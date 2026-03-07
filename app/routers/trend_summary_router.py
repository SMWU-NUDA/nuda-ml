# app/routers/trend_summary_router.py
from typing import List, Optional
import os
import re

from fastapi import APIRouter, HTTPException, Path
from pydantic import BaseModel
from psycopg import connect
from psycopg.rows import dict_row

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


class TrendSummaryRes(BaseModel):
    productId: int
    totalReviewCount: int
    trendHighlights: List[str]
    analyzedAt: Optional[str] = None


_BULLET_PREFIX = re.compile(r"^\s*(?:[•\-\*]|(\d+)[\.\)])\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_KO_EOS = ["습니다", "입니다", "합니다", "됩니다", "있습니다", "없습니다"]

def _split_by_korean_eos(t: str) -> List[str]:
    out: List[str] = []
    buf = ""

    i = 0
    n = len(t)
    while i < n:
        buf += t[i]
        for eos in _KO_EOS:
            if buf.endswith(eos):
                j = i + 1
                if j < n and t[j] == ".":
                    buf += "."
                    i += 1
                out.append(buf.strip())
                buf = ""
                break
        i += 1

    tail = buf.strip()
    if tail:
        out.append(tail)
    return out

def split_to_highlights(text: str, max_items: int = 3) -> List[str]:
    if not text:
        return []

    t = " ".join(text.strip().split())

    parts = [p.strip() for p in _SENT_SPLIT.split(t) if p.strip()]

    if len(parts) <= 1:
        parts = [p.strip() for p in _split_by_korean_eos(t) if p.strip()]

    parts = [p for p in parts if len(p) >= 6]

    return parts[:max_items] if parts else [t]

def split_summary_to_bullets(summary: str) -> List[str]:
    if not summary:
        return []
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    if not lines:
        return []

    bullets: List[str] = []
    for ln in lines:
        ln2 = _BULLET_PREFIX.sub("", ln).strip()
        if ln2:
            bullets.append(ln2)

    if len(bullets) == 1 and "•" in bullets[0]:
        parts = [p.strip() for p in bullets[0].split("•") if p.strip()]
        if len(parts) >= 2:
            bullets = parts

    if len(bullets) == 1:
        bullets = split_to_highlights(bullets[0], max_items=3)

    return bullets


REVIEW_COUNT_SQL = """
SELECT review_count FROM public.product WHERE id = %(product_id)s LIMIT 1
"""


@router.get(
    "/{product_id}/review-trend",
    response_model=TrendSummaryRes,
    summary="상품별 리뷰 트렌드 요약 API",
)
def get_trend_summary(
    product_id: int = Path(..., ge=1, description="상품 ID")
):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # 리뷰 수 체크
                cur.execute(REVIEW_COUNT_SQL, {"product_id": product_id})
                prow = cur.fetchone()
                if prow is not None and int(prow.get("review_count") or 0) < 10:
                    raise HTTPException(status_code=422, detail="리뷰수가 10개 이하입니다.")

                sql = """
                WITH rs AS (
                  SELECT
                    s.product_id,
                    s.summary,
                    s.updated_at
                  FROM ai_review_summary s
                  WHERE s.product_id = %(product_id)s
                  ORDER BY s.updated_at DESC
                  LIMIT 1
                ),
                rc AS (
                  SELECT
                    r.product_id,
                    COUNT(*)::int AS total_review_count
                  FROM review r
                  WHERE r.product_id = %(product_id)s
                  GROUP BY r.product_id
                )
                SELECT
                  rs.product_id,
                  COALESCE(rc.total_review_count, 0) AS total_review_count,
                  rs.summary,
                  rs.updated_at
                FROM rs
                LEFT JOIN rc ON rc.product_id = rs.product_id;
                """
                cur.execute(sql, {"product_id": product_id})
                row = cur.fetchone()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not row:
        raise HTTPException(status_code=404, detail="trend summary not found")

    bullets = split_summary_to_bullets(row["summary"] or "")
    items = bullets[:3] if bullets else split_to_highlights(row["summary"] or "", max_items=3)

    return {
        "productId": row["product_id"],
        "totalReviewCount": row["total_review_count"],
        "trendHighlights": items,
        "analyzedAt": row["updated_at"].isoformat() if row["updated_at"] else None,
    }