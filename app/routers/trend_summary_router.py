# app/routers/trend_summary_router.py
from typing import List, Optional
import os
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
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

# ===== Request / Response =====

class TrendSummaryReq(BaseModel):
    productId: int = Field(..., ge=1)

class TrendSummaryItem(BaseModel):
    content: str

class TrendSummaryRes(BaseModel):
    productId: int
    totalReviewCount: int
    trendHighlights: List[str]
    analyzedAt: str

# ===== Helpers =====

_BULLET_PREFIX = re.compile(r"^\s*(?:[•\-\*]|(\d+)[\.\)])\s+")

_PUNCT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+")
# 마침표가 없어도 "습니다/합니다/됩니다/있습니다..." 같은 종결에서 자르기
_KO_EOS = re.compile(r"(습니다|입니다|합니다|됩니다|있습니다|없습니다)\s*")

def split_to_highlights(text: str, max_items: int = 3) -> List[str]:
    if not text:
        return []
    t = " ".join(text.strip().split())  # 공백 정리

    # 1) 문장부호 있으면 우선 사용
    parts = [p.strip() for p in _PUNCT_SPLIT.split(t) if p.strip()]
    if len(parts) >= 2:
        return parts[:max_items]

    # 2) 한국어 종결어미 기반으로 자르기 (마침표 없어도)
    cuts = [m.end() for m in _KO_EOS.finditer(t)]
    if cuts:
        out = []
        start = 0
        for end in cuts:
            seg = t[start:end].strip()
            if seg:
                out.append(seg)
            start = end
            if len(out) >= max_items:
                break
        if len(out) < max_items:
            tail = t[start:].strip()
            if tail:
                out.append(tail)
        # 너무 짧은 조각 제거
        out = [s for s in out if len(s) >= 8]
        return out[:max_items] if out else [t]

    # 3) fallback: 그냥 1개
    return [t]

def split_summary_to_bullets(summary: str) -> List[str]:
    """
    summary 텍스트가
    - "• 문장\n• 문장"
    - "- 문장\n- 문장"
    - "1. 문장\n2. 문장"
    - 그냥 여러 줄
    같은 형태일 수 있어서 최대한 안전하게 줄 단위로 bullets 추출
    """
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

    # 만약 줄 분리가 안 먹고 한 줄에 "•"가 여러 개면 추가 분해
    if len(bullets) == 1 and "•" in bullets[0]:
        parts = [p.strip() for p in bullets[0].split("•") if p.strip()]
        if len(parts) >= 2:
            bullets = parts

    return bullets

# ===== Endpoint =====

@router.post("/{product_id}/review-trend", response_model=TrendSummaryRes
             ,summary="상품별 리뷰 트렌드 요약 API",
             description="상품별로 최신 리뷰 요약과 총 리뷰수를 반환합니다.\n\n")
def get_trend_summary(req: TrendSummaryReq):
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

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"product_id": req.productId})
                row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if not row:
        raise HTTPException(status_code=404, detail="trend summary not found")

    items = split_to_highlights(row["summary"] or "", max_items=3)
    trend_items = [{"content": s} for s in items]



    items = split_to_highlights(row["summary"] or "", max_items=3)
    return {
        "productId": row["product_id"],
        "totalReviewCount": row["total_review_count"],
        "trendHighlights": items,
        "analyzedAt": row["updated_at"].isoformat() if row["updated_at"] else None, 
        }
