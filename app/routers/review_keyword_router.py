from typing import Optional, Literal, List, Dict, Any
import os

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field
import psycopg
from psycopg.rows import dict_row

router = APIRouter(prefix="/ml/reviews", tags=["reviews"])

def get_conn():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        row_factory=dict_row,
        connect_timeout=10,
    )

Keyword = Literal["default", "irritationLevel", "scent", "adhesion", "absorption"]

KEYWORD_TO_FEATURE_KEY = {
    "irritationLevel": "민감/자극",
    "scent": "향/냄새",
    "absorption": "흡수/샘",
    "adhesion": "접착/고정",
}
FEATURE_KEYS = ["민감/자극", "향/냄새", "흡수/샘", "접착/고정"]

# =========================
# 기존: ID만 반환
# =========================
class RankedIdsResp(BaseModel):
    keyword: Keyword = Field(description="default: 전체")
    rankedIds: List[int]

@router.get(
    "/{product_id}",
    response_model=RankedIdsResp,
    summary="상품별 리뷰 키워드 기반 조회",
)
def list_reviews(
    product_id: int,
    keyword: Keyword = Query(default="default"),
    topK: int = Query(default=300, ge=1, le=500),
):
    params: Dict[str, Any] = {"product_id": product_id, "limit": topK}

    if keyword == "default":
        sql = """
        WITH smax AS (
          SELECT product_id, review_id, MAX(score) AS score
          FROM review_feature_score
          WHERE product_id = %(product_id)s
            AND feature_key = ANY(%(feature_keys)s)
          GROUP BY product_id, review_id
        )
        SELECT r.id AS review_id
        FROM review r
        LEFT JOIN smax
          ON smax.product_id = r.product_id
         AND smax.review_id  = r.id
        WHERE r.product_id = %(product_id)s
        ORDER BY COALESCE(smax.score, 0) DESC, r.id DESC
        LIMIT %(limit)s;
        """
        params["feature_keys"] = FEATURE_KEYS
    else:
        feature_key = KEYWORD_TO_FEATURE_KEY.get(keyword)
        if not feature_key:
            raise HTTPException(status_code=400, detail="Invalid keyword")

        params["feature_key"] = feature_key
        sql = """
        SELECT r.id AS review_id
        FROM review r
        JOIN review_feature_score s
          ON s.product_id = r.product_id
         AND s.review_id  = r.id
         AND s.feature_key = %(feature_key)s
        WHERE r.product_id = %(product_id)s
          AND s.score > 0
        ORDER BY s.score DESC, r.id DESC
        LIMIT %(limit)s;
        """

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    ranked_ids = [int(r["review_id"]) for r in rows]
    return RankedIdsResp(keyword=keyword, rankedIds=ranked_ids)

class RankedReviewItem(BaseModel):
    reviewId: int
    content: Optional[str] = None
    score: float = 0.0
    featureKey: Optional[str] = None

class RankedReviewsDebugResp(BaseModel):
    keyword: Keyword = Field(description="default: 전체")
    topK: int
    reviews: List[RankedReviewItem]

@router.get(
    "/{product_id}/debug",
    response_model=RankedReviewsDebugResp,
    summary="(디버깅용) 상품별 리뷰 키워드 기반 조회",
)
def list_reviews_debug(
    product_id: int,
    keyword: Keyword = Query(default="default"),
    topK: int = Query(default=50, ge=1, le=200),
):
    params: Dict[str, Any] = {"product_id": product_id, "limit": topK}


    REVIEW_TEXT_COL = "r.content"

    if keyword == "default":
        sql = f"""
        WITH smax AS (
          SELECT product_id, review_id, MAX(score) AS score
          FROM review_feature_score
          WHERE product_id = %(product_id)s
            AND feature_key = ANY(%(feature_keys)s)
          GROUP BY product_id, review_id
        )
        SELECT
          r.id AS review_id,
          {REVIEW_TEXT_COL} AS content,
          COALESCE(smax.score, 0) AS score,
          NULL::text AS feature_key
        FROM review r
        LEFT JOIN smax
          ON smax.product_id = r.product_id
         AND smax.review_id  = r.id
        WHERE r.product_id = %(product_id)s
        ORDER BY COALESCE(smax.score, 0) DESC, r.id DESC
        LIMIT %(limit)s;
        """
        params["feature_keys"] = FEATURE_KEYS
    else:
        feature_key = KEYWORD_TO_FEATURE_KEY.get(keyword)
        if not feature_key:
            raise HTTPException(status_code=400, detail="Invalid keyword")

        params["feature_key"] = feature_key
        sql = f"""
        SELECT
          r.id AS review_id,
          {REVIEW_TEXT_COL} AS content,
          s.score AS score,
          s.feature_key AS feature_key
        FROM review r
        JOIN review_feature_score s
          ON s.product_id = r.product_id
         AND s.review_id  = r.id
         AND s.feature_key = %(feature_key)s
        WHERE r.product_id = %(product_id)s
          AND s.score > 0
        ORDER BY s.score DESC, r.id DESC
        LIMIT %(limit)s;
        """

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    reviews = [
        RankedReviewItem(
            reviewId=int(r["review_id"]),
            content=r.get("content"),
            score=float(r.get("score") or 0),
            featureKey=r.get("feature_key"),
        )
        for r in rows
    ]

    return RankedReviewsDebugResp(keyword=keyword, topK=topK, reviews=reviews)