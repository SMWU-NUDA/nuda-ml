# app/routers/review_keyword_router.py
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


Keyword = Literal["all", "irritationLevel", "scent", "adhesion", "absorption"]

# UI keyword -> DB feature_key (review_feature_score.feature_key)
KEYWORD_TO_FEATURE_KEY = {
    "irritationLevel": "민감/자극",
    "scent": "향/냄새",
    "absorption": "흡수/샘",
    "adhesion": "접착/고정",
}

FEATURE_KEYS = ["민감/자극", "향/냄새", "흡수/샘", "접착/고정"]


# =========================
# GET: 상품별 리뷰 조회 (점수 기준 정렬 + 무한스크롤)
# =========================
class Cursor(BaseModel):
    score: float
    reviewId: int


class ReviewItem(BaseModel):
    reviewId: int
    content: str
    rating: Optional[float] = None
    createdAt: Optional[str] = None
    likeCount: Optional[int] = None
    score: float


class ReviewListResp(BaseModel):
    items: List[ReviewItem]
    nextCursor: Optional[Cursor] = None


@router.get(
    "/{product_id}",
    response_model=ReviewListResp,
    summary="상품별 리뷰조회시, 키워드 기반 필터링 조회 API",
)
def list_reviews(
    product_id: int,
    keyword: Keyword = Query(default="all"),
    limit: int = Query(default=20, ge=1, le=100),
    cursorScore: Optional[float] = Query(default=None),
    cursorReviewId: Optional[int] = Query(default=None),
):
    # cursor 없으면 SQL에서 cursor 조건 자체를 빼야 psycopg/pg 타입 애매함 에러가 안남
    use_cursor = (cursorScore is not None) and (cursorReviewId is not None)

    params: Dict[str, Any] = {"product_id": product_id, "limit": limit}
    if use_cursor:
        params["cursor_score"] = float(cursorScore)
        params["cursor_review_id"] = int(cursorReviewId)

    if keyword == "all":
        sql = """
        WITH smax AS (
          SELECT product_id, review_id, MAX(score) AS score
          FROM review_feature_score
          WHERE product_id = %(product_id)s
          GROUP BY product_id, review_id
        )
        SELECT
          r.id AS review_id,
          r.content,
          r.rating,
          r.created_at,
          r.like_count,
          COALESCE(smax.score, 0) AS score
        FROM review r
        LEFT JOIN smax
          ON smax.product_id = r.product_id
         AND smax.review_id  = r.id
        WHERE r.product_id = %(product_id)s
        """
        if use_cursor:
            sql += """
              AND (COALESCE(smax.score, 0), r.id) < (%(cursor_score)s, %(cursor_review_id)s)
            """
        sql += """
        ORDER BY COALESCE(smax.score, 0) DESC, r.id DESC
        LIMIT %(limit)s;
        """
    else:
        feature_key = KEYWORD_TO_FEATURE_KEY[keyword]
        params["feature_key"] = feature_key

        sql = """
        SELECT
          r.id AS review_id,
          r.content,
          r.rating,
          r.created_at,
          r.like_count,
          s.score AS score
        FROM review r
        JOIN review_feature_score s
          ON s.product_id = r.product_id
         AND s.review_id  = r.id
         AND s.feature_key = %(feature_key)s
        WHERE r.product_id = %(product_id)s
          AND s.score > 0
        """
        if use_cursor:
            sql += """
              AND (s.score, r.id) < (%(cursor_score)s, %(cursor_review_id)s)
            """
        sql += """
        ORDER BY s.score DESC, r.id DESC
        LIMIT %(limit)s;
        """

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    items: List[ReviewItem] = []
    for row in rows:
        created_at = row.get("created_at")
        items.append(
            ReviewItem(
                reviewId=int(row["review_id"]),
                content=row.get("content") or "",
                rating=row.get("rating"),
                createdAt=created_at.isoformat() if created_at else None,
                likeCount=row.get("like_count"),
                score=float(row.get("score") or 0.0),
            )
        )

    next_cursor = None
    if rows:
        last = rows[-1]
        next_cursor = Cursor(score=float(last.get("score") or 0.0), reviewId=int(last["review_id"]))

    return ReviewListResp(items=items, nextCursor=next_cursor)


# =========================
# POST: 신규 리뷰 점수 계산 + DB upsert
# =========================
class ReviewIn(BaseModel):
    reviewId: int
    content: str
    rating: Optional[float] = None
    createdAt: Optional[str] = None


class ReviewScoreReq(BaseModel):
    keyword: Keyword = Field(default="all")
    reviews: List[ReviewIn]


class ScoreOut(BaseModel):
    reviewId: int
    score: float


class ReviewScoreResp(BaseModel):
    scores: List[ScoreOut]


def _normalize_text(s: str) -> str:
    return (s or "").strip()


def _score_review_by_dict(content: str, kw_dict: Dict[str, Any]) -> float:
    text = _normalize_text(content)
    if not text or not isinstance(kw_dict, dict) or not kw_dict:
        return 0.0

    best = 0.0
    for phrase, w in kw_dict.items():
        if not phrase:
            continue
        if phrase in text:
            try:
                fw = float(w)
            except Exception:
                continue
            if fw > best:
                best = fw

    if best < 0.0:
        return 0.0
    if best > 1.0:
        return 1.0
    return float(best)


SQL_GET_FEATURE_SCORES = """
SELECT feature_scores
FROM product_review_keywords
WHERE product_id = %(product_id)s
LIMIT 1;
"""

SQL_UPSERT_RFS = """
INSERT INTO review_feature_score (
  product_id, review_id, feature_key, score, matched, review_updated_at, computed_at
)
VALUES (
  %(product_id)s, %(review_id)s, %(feature_key)s, %(score)s, %(matched)s::jsonb, now(), now()
)
ON CONFLICT (product_id, review_id, feature_key)
DO UPDATE SET
  score = EXCLUDED.score,
  matched = EXCLUDED.matched,
  review_updated_at = EXCLUDED.review_updated_at,
  computed_at = now();
"""


@router.post(
    "/{product_id}/review-scores",
    response_model=ReviewScoreResp,
    summary="신규 리뷰 점수 계산 후 review_feature_score에 upsert",
)
def compute_and_upsert_review_scores(product_id: int, body: ReviewScoreReq):
    with get_conn() as conn:
        row = conn.execute(SQL_GET_FEATURE_SCORES, {"product_id": product_id}).fetchone()

        if not row or row["feature_scores"] is None:
            # 사전(feature_scores) 없으면 전부 0점 반환 (DB에 굳이 저장하지 않음)
            return {"scores": [{"reviewId": r.reviewId, "score": 0.0} for r in body.reviews]}

        feature_scores: Dict[str, Any] = row["feature_scores"]

        scores_out: List[Dict[str, Any]] = []

        if body.keyword == "all":
            # all 요청이면 4개 feature_key 모두 계산해서 저장
            for r in body.reviews:
                max_score = 0.0
                for fk in FEATURE_KEYS:
                    d = feature_scores.get(fk, {})
                    s = _score_review_by_dict(r.content, d)

                    conn.execute(
                        SQL_UPSERT_RFS,
                        {
                            "product_id": product_id,
                            "review_id": r.reviewId,
                            "feature_key": fk,
                            "score": s,
                            "matched": None,
                        },
                    )
                    if s > max_score:
                        max_score = s

                scores_out.append({"reviewId": r.reviewId, "score": float(max_score)})

            conn.commit()
            return {"scores": scores_out}

        # keyword != all
        fk = KEYWORD_TO_FEATURE_KEY.get(body.keyword)
        if not fk:
            raise HTTPException(status_code=400, detail="Invalid keyword")

        kw_dict = feature_scores.get(fk, {})

        for r in body.reviews:
            s = _score_review_by_dict(r.content, kw_dict)

            conn.execute(
                SQL_UPSERT_RFS,
                {
                    "product_id": product_id,
                    "review_id": r.reviewId,
                    "feature_key": fk,
                    "score": s,
                    "matched": None,
                },
            )
            scores_out.append({"reviewId": r.reviewId, "score": float(s)})

        conn.commit()
        return {"scores": scores_out}
