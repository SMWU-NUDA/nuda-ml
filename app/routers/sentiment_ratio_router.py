# app/routers/sentiment_router.py

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel
from psycopg import connect
from psycopg.rows import dict_row
import os

router = APIRouter(prefix="/ml/products", tags=["reviews"])

def get_conn():
    return connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        row_factory=dict_row,
    )

class SentimentDistribution(BaseModel):
    positive: float
    negative: float

class SentimentRes(BaseModel):
    productId: int
    sentimentDistribution: SentimentDistribution
    totalCount: int
    analyzedAt: Optional[str] = None

REVIEW_COUNT_SQL = """
SELECT review_count FROM public.product WHERE id = %(product_id)s LIMIT 1
"""

@router.get(
    "/{product_id}/review-sentiment",
    response_model=SentimentRes,
    summary="상품별 리뷰 감성 비율 API",
    description="상품에 대한 긍정/부정 리뷰 비율과 총 리뷰 수를 반환하는 API입니다.",
)
def get_sentiment_summary(product_id: int = Path(..., ge=1)):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # 리뷰 수 체크
                cur.execute(REVIEW_COUNT_SQL, {"product_id": product_id})
                prow = cur.fetchone()
                if prow is not None and int(prow.get("review_count") or 0) < 10:
                    raise HTTPException(status_code=422, detail="리뷰수가 10개 미만입니다.")

                sql = """
                SELECT
                    s.product_id,
                    s.pos_ratio,
                    s.neg_ratio,
                    p.review_count,
                    rs.updated_at
                FROM product_review_summary s
                JOIN product p
                  ON p.id = s.product_id
                LEFT JOIN ai_review_summary rs
                  ON rs.product_id = s.product_id
                WHERE s.product_id = %(product_id)s
                ORDER BY rs.updated_at DESC
                LIMIT 1
                """
                cur.execute(sql, {"product_id": product_id})
                row = cur.fetchone()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        raise HTTPException(status_code=404, detail="no sentiment data")

    updated_at = row["updated_at"]
    return {
        "productId": row["product_id"],
        "sentimentDistribution": {
            "positive": float(row["pos_ratio"]),
            "negative": float(row["neg_ratio"]),
        },
        "totalCount": int(row["review_count"]),
        "analyzedAt": updated_at.isoformat() if updated_at else None,
    }