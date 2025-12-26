#!/usr/bin/env python3
"""
Strategy Research Lab REST API

TradingView 전략 분석 결과를 제공하는 FastAPI 서버
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional, List, Any
from datetime import datetime

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ============================================================
# Pydantic Models
# ============================================================

class StatsResponse(BaseModel):
    """통계 정보 응답"""
    total_strategies: int = Field(..., description="총 전략 수")
    analyzed_count: int = Field(..., description="분석 완료 수")
    passed_count: int = Field(..., description="권장 전략 수 (A, B 등급)")
    avg_score: float = Field(..., description="평균 점수")


class StrategyItem(BaseModel):
    """전략 목록 아이템"""
    script_id: str
    title: str
    author: str
    likes: int
    total_score: Optional[float] = None
    grade: Optional[str] = None
    repainting_score: Optional[float] = None
    overfitting_score: Optional[float] = None


class StrategyDetail(BaseModel):
    """전략 상세 정보"""
    script_id: str
    title: str
    author: str
    likes: int
    total_score: Optional[float] = None
    grade: Optional[str] = None
    repainting_score: Optional[float] = None
    overfitting_score: Optional[float] = None
    pine_code: Optional[str] = None
    pine_version: Optional[int] = None
    performance: Optional[dict] = None
    analysis: Optional[dict] = None
    created_at: Optional[str] = None


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Strategy Research Lab API",
    description="TradingView 전략 분석 결과 API - Hetzner 자동 배포",
    version="1.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 경로 설정 (서버 배포 기준)
BASE_DIR = Path("/opt/strategy-research-lab")
DB_PATH = BASE_DIR / "data" / "strategies.db"
DATA_DIR = BASE_DIR / "data"


def get_db():
    """데이터베이스 연결"""
    if not DB_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Database not found: {DB_PATH}"
        )
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def parse_json_field(value: Any) -> Optional[dict]:
    """JSON 필드 파싱"""
    if not value:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_analysis_data(analysis_json: str) -> dict:
    """analysis_json에서 점수 및 등급 추출"""
    analysis = parse_json_field(analysis_json)
    if not analysis:
        return {
            "total_score": None,
            "grade": None,
            "repainting_score": None,
            "overfitting_score": None
        }

    return {
        "total_score": analysis.get("total_score"),
        "grade": analysis.get("grade"),
        "repainting_score": analysis.get("repainting_score"),
        "overfitting_score": analysis.get("overfitting_score")
    }


# ============================================================
# API Endpoints
# ============================================================

@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_exists": DB_PATH.exists()
    }


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """통계 정보 조회"""
    try:
        conn = get_db()
        cur = conn.cursor()

        # 총 전략 수
        cur.execute("SELECT COUNT(*) FROM strategies")
        total = cur.fetchone()[0]

        # analysis_json이 있는 항목 수 (분석 완료)
        cur.execute("SELECT COUNT(*) FROM strategies WHERE analysis_json IS NOT NULL AND analysis_json != ''")
        analyzed = cur.fetchone()[0]

        # 모든 분석된 전략의 analysis_json 가져와서 통계 계산
        cur.execute("SELECT analysis_json FROM strategies WHERE analysis_json IS NOT NULL AND analysis_json != ''")
        rows = cur.fetchall()
        conn.close()

        passed = 0
        total_score_sum = 0
        score_count = 0

        for row in rows:
            data = extract_analysis_data(row[0])
            grade = data.get("grade")
            score = data.get("total_score")

            if grade in ('A', 'B'):
                passed += 1
            if score is not None:
                total_score_sum += score
                score_count += 1

        avg_score = total_score_sum / score_count if score_count > 0 else 0

        return StatsResponse(
            total_strategies=total,
            analyzed_count=analyzed,
            passed_count=passed,
            avg_score=round(avg_score, 1)
        )

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/strategies", response_model=List[StrategyItem])
async def get_strategies(
    limit: int = Query(50, ge=1, le=200, description="조회 개수"),
    offset: int = Query(0, ge=0, description="오프셋"),
    min_score: float = Query(0, ge=0, le=100, description="최소 점수"),
    grade: Optional[str] = Query(None, description="등급 필터 (A, B, C, D, F)"),
    search: Optional[str] = Query(None, description="검색어 (제목, 작성자)"),
    sort_by: str = Query("likes", description="정렬 기준 (likes, title)"),
    sort_order: str = Query("desc", description="정렬 순서 (asc, desc)")
):
    """전략 목록 조회"""
    try:
        conn = get_db()
        cur = conn.cursor()

        # 기본 쿼리 - analysis_json이 있는 전략만
        query = """
            SELECT script_id, title, author, likes, analysis_json
            FROM strategies
            WHERE analysis_json IS NOT NULL AND analysis_json != ''
        """
        params: List = []

        # 검색
        if search:
            query += " AND (title LIKE ? OR author LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        # 정렬 (DB 컬럼 기준)
        valid_columns = ["likes", "title", "created_at"]
        if sort_by not in valid_columns:
            sort_by = "likes"
        order = "DESC" if sort_order.lower() == "desc" else "ASC"
        query += f" ORDER BY {sort_by} {order}"

        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

        # 결과 가공 (JSON 파싱 + 필터링)
        results = []
        for row in rows:
            data = extract_analysis_data(row["analysis_json"])
            total_score = data.get("total_score") or 0
            strategy_grade = data.get("grade")

            # 필터 적용
            if total_score < min_score:
                continue
            if grade and strategy_grade != grade:
                continue

            results.append(StrategyItem(
                script_id=row["script_id"],
                title=row["title"] or "",
                author=row["author"] or "",
                likes=row["likes"] or 0,
                total_score=data.get("total_score"),
                grade=strategy_grade,
                repainting_score=data.get("repainting_score"),
                overfitting_score=data.get("overfitting_score")
            ))

        # 점수 기준 정렬 (클라이언트 요청 시)
        if sort_by == "score" or sort_by == "total_score":
            results.sort(key=lambda x: x.total_score or 0, reverse=(sort_order.lower() == "desc"))

        # 페이징
        return results[offset:offset + limit]

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/api/strategy/{script_id}", response_model=StrategyDetail)
async def get_strategy_detail(script_id: str):
    """전략 상세 정보 조회"""
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM strategies WHERE script_id = ?", [script_id])
        row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Strategy not found")

        # JSON 필드 파싱
        performance = parse_json_field(row["performance_json"])
        analysis = parse_json_field(row["analysis_json"])
        analysis_data = extract_analysis_data(row["analysis_json"])

        return StrategyDetail(
            script_id=row["script_id"],
            title=row["title"] or "",
            author=row["author"] or "",
            likes=row["likes"] or 0,
            total_score=analysis_data.get("total_score"),
            grade=analysis_data.get("grade"),
            repainting_score=analysis_data.get("repainting_score"),
            overfitting_score=analysis_data.get("overfitting_score"),
            pine_code=row["pine_code"],
            pine_version=row["pine_version"],
            performance=performance,
            analysis=analysis,
            created_at=row["created_at"]
        )

    except HTTPException:
        raise
    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# ============================================================
# Static Files & Main Page
# ============================================================

@app.get("/")
async def serve_index():
    """메인 페이지 (초보자 리포트)"""
    html_file = DATA_DIR / "beginner_report.html"
    if html_file.exists():
        return FileResponse(html_file)
    return {"message": "Welcome to Strategy Research Lab API", "docs": "/api/docs"}


@app.get("/report.html")
async def serve_report():
    """일반 리포트"""
    html_file = DATA_DIR / "report.html"
    if html_file.exists():
        return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="Report not found")


# ============================================================
# Backtest Endpoints
# ============================================================

class BacktestRequest(BaseModel):
    """백테스트 요청"""
    script_id: str = Field(..., description="전략 ID")
    symbol: str = Field("BTC/USDT", description="거래쌍")
    timeframe: str = Field("1h", description="시간프레임")
    start_date: str = Field("2024-01-01", description="시작일")
    end_date: str = Field("2024-12-01", description="종료일")
    initial_capital: float = Field(10000.0, description="초기 자본")


class BacktestResponse(BaseModel):
    """백테스트 응답"""
    success: bool
    script_id: str
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    backtest: Optional[dict] = None
    error: Optional[str] = None
    tested_at: Optional[str] = None


@app.post("/api/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    전략 백테스트 실행

    Pine Script 전략을 Python으로 변환하고 과거 데이터로 백테스트합니다.
    """
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))

    try:
        from backtester import StrategyTester

        tester = StrategyTester(str(DB_PATH))

        result = await tester.test_strategy(
            script_id=request.script_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital
        )

        if result.get('error'):
            return BacktestResponse(
                success=False,
                script_id=request.script_id,
                error=result['error']
            )

        return BacktestResponse(
            success=True,
            script_id=request.script_id,
            symbol=request.symbol,
            timeframe=request.timeframe,
            backtest=result.get('backtest'),
            tested_at=result.get('tested_at')
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


@app.post("/api/backtest/all")
async def run_all_backtests(
    limit: int = Query(5, ge=1, le=50, description="테스트할 전략 수"),
    symbol: str = Query("BTC/USDT", description="거래쌍"),
    timeframe: str = Query("1h", description="시간프레임"),
    start_date: str = Query("2024-01-01", description="시작일"),
    end_date: str = Query("2024-06-01", description="종료일")
):
    """
    모든 전략 백테스트 (상위 N개)

    Pine Script 코드가 있는 전략을 좋아요 순으로 정렬하여 백테스트합니다.
    """
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))

    try:
        from backtester import StrategyTester

        tester = StrategyTester(str(DB_PATH))

        results = await tester.test_all_strategies(
            limit=limit,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        # 요약 통계
        successful = [r for r in results if r.get('backtest', {}).get('success')]
        total_return_sum = sum(r['backtest']['total_return'] for r in successful) if successful else 0
        avg_return = total_return_sum / len(successful) if successful else 0

        return {
            "total_tested": len(results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "avg_return": round(avg_return, 2),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest error: {str(e)}")


@app.get("/api/strategy/{script_id}/backtest")
async def get_backtest_result(script_id: str):
    """전략의 저장된 백테스트 결과 조회"""
    try:
        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT analysis_json FROM strategies WHERE script_id = ?", [script_id])
        row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Strategy not found")

        analysis = parse_json_field(row[0])

        if not analysis or 'backtest_result' not in analysis:
            return {
                "script_id": script_id,
                "has_backtest": False,
                "message": "No backtest result available. Run /api/backtest first."
            }

        return {
            "script_id": script_id,
            "has_backtest": True,
            "backtest_result": analysis['backtest_result']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# 정적 파일 마운트 (맨 마지막에 배치)
if DATA_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DATA_DIR)), name="static")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
