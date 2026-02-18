# app/routers/sentiment_ratio_router.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from psycopg import connect
from psycopg.rows import dict_row
import os

router = APIRouter(prefix="/ml/products", tags=["reviews"])

# DB 연결
def get_conn():
    return connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        row_factory=dict_row,
    )


# Request
class SentimentReq(BaseModel):
    productId: int = Field(...)


# Response
class SentimentDistribution(BaseModel):
    positive: float
    negative: float


class SentimentRes(BaseModel):

    productId: int

    sentimentDistribution: SentimentDistribution

    totalCount: int

    analyzedAt: str


# endpoint
@router.post("/{product_id}/review-sentiment", response_model=SentimentRes,
             summary="상품별 리뷰 감성 비율 API",
             description="상품에 대한 긍정/부정 리뷰 비율과 총 리뷰 수를 반환하는 API입니다.")
def get_sentiment_summary(req: SentimentReq):

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

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"product_id": req.productId})
                row = cur.fetchone()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        raise HTTPException(status_code=404, detail="no sentiment data")


    positive = row["pos_ratio"]
    negative = row["neg_ratio"]


    return {

        "productId": row["product_id"],

        "sentimentDistribution": {

            "positive": row["pos_ratio"],
            "negative": row["neg_ratio"]
        },

        "totalCount": row["review_count"],

        "analyzedAt":
            row["updated_at"].isoformat()
            if row["updated_at"]
            else None
    }