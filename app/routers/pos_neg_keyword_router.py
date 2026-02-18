# app/routers/pos_neg_keyword_router.py

import os
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from typing import List, Dict, Optional, Any
from psycopg.errors import UndefinedColumn

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row

router = APIRouter(prefix="/ml/products", tags=["reviews"])

# =======================
# DB
# =======================
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

def _coerce_json(v: Any) -> Any:
    """jsonb 컬럼이 dict/list로 오거나 문자열(JSON)로 오는 케이스 모두 처리"""
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None

# =======================
# Keyword cleaning
# =======================
STOPWORDS = {
    "제품","구매","재구매","사용","쓰고","써봤","써봄","구매했","구매했는데",
    "괜찮","괜찮음","괜찮아요","좋아","좋음","좋아요","좋다","만족","추천",
    "느낌","개인적","여러","브랜드","이번","이전","다시","같아서","지금",
    "배송","포장","가격","처음","계속","항상","정말","완전","진짜","너무",
    "그리고","바로","가장","그치","요기","이제품"
}

SUFFIXES = (
    "입니다","해요","했어요","했는데","같아요","같음","네요","어요","아요",
    "은","는","이","가","을","를","에","에서","와","과","도","만","까지","부터","으로","로"
)

def normalize_kw(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"\s+", " ", s)

    # 한글/공백만 남기기
    s = re.sub(r"[^가-힣\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if len(s) <= 1 or len(s) >= 25:
        return ""

    # suffix strip 최대 2회
    for _ in range(2):
        changed = False
        for suf in SUFFIXES:
            if s.endswith(suf) and len(s) - len(suf) >= 2:
                s = s[: -len(suf)].strip()
                changed = True
        if not changed:
            break

    if len(s) <= 1:
        return ""

    if s in STOPWORDS:
        return ""

    # 너무 문장형인 것 약 컷
    bad_contains = ["해보겠", "혜택", "구매해", "구입해", "듭니", "사용하"]
    if any(b in s for b in bad_contains):
        return ""

    return s

# =======================
# Positive: feature_scores -> topn keywords (score 기반)
# =======================
def pack_feature_scores(feature_scores: Dict[str, Dict[str, float]], topn: int = 5) -> List[Dict[str, str]]:
    candidates: Dict[str, float] = {}

    for _aspect, kw_map in (feature_scores or {}).items():
        if not isinstance(kw_map, dict):
            continue
        for k, s in kw_map.items():
            k2 = normalize_kw(k)
            if not k2:
                continue
            try:
                score = float(s)
            except Exception:
                continue
            prev = candidates.get(k2)
            candidates[k2] = score if prev is None else max(prev, score)

    if not candidates:
        return []

    items = list(candidates.items())
    items.sort(key=lambda t: (-t[1], len(t[0]), t[0]))  # score desc
    return [{"keyword": k} for k, _ in items[:topn]]

# =======================
# Positive beautify (짧은 자연어)
# =======================
POSITIVE_PATTERNS = [
    # 민감/자극
    (["트러블"], "트러블이 나지 않음"),
    (["자극"], "자극 없음"),
    (["가려움", "간지러"], "가려움 없음"),
    (["쓸림"], "쓸림 없음"),
    (["피부"], "피부에 순함"),

    # 흡수/샘
    (["흡수력"], "흡수력 좋음"),
    (["흡수"], "흡수 잘 됨"),
    (["샘", "새지"], "새지 않음"),

    # 착용감/고정
    (["부드", "촉감"], "부드러움"),
    (["편하", "착용감"], "착용감 편함"),
    (["밀착"], "밀착력 좋음"),
    (["고정", "접착"], "고정력 좋음"),

    # 길이/두께
    (["롱라이너"], "롱라이너라 안심"),
    (["길이", "롱"], "길이 충분함"),
    (["도톰"], "도톰함"),
    (["두께"], "두께 적당함"),
    (["얇"], "얇아서 편함"),

    # 가격/가성비
    (["가성비"], "가성비 좋음"),
    (["가격"], "가격 합리적"),
    (["할인"], "할인 시 만족"),

    # 향/냄새
    (["무향"], "무향"),
    (["냄새"], "냄새 없음"),
]

def beautify_positive(keyword: str) -> str:
    for keys, label in POSITIVE_PATTERNS:
        if any(k in keyword for k in keys):
            return label
    return keyword  # fallback: 원문 그대로

def beautify_positive_list(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for it in items:
        k = (it.get("keyword") or "").strip()
        if not k:
            continue
        out.append({"keyword": beautify_positive(k)})
    return out

# =======================
# Negative: ai_review_summary.summary -> short labels
# =======================
NEG_SHORT_PATTERNS = [
    (["답답"], "답답함"),
    (["덥"], "더움"),
    (["습", "찝찝"], "찝찝함"),

    (["자극"], "자극 있음"),
    (["간지", "가렵"], "간지러움"),
    (["트러블"], "트러블 발생"),

    (["샘", "새"], "샘 있음"),
    (["흡수"], "흡수력 약함"),

    (["냄새"], "냄새 있음"),
    (["향"], "향이 강함"),

    (["두껍", "도톰"], "두꺼움"),

    (["개인차"], "개인차 있음"),
    (["주의"], "주의 필요"),
]

def ai_summary_to_short_negatives(summary: str, topn: int = 5) -> List[Dict[str, str]]:
    if not summary:
        return []

    found: List[str] = []
    for keys, label in NEG_SHORT_PATTERNS:
        if any(k in summary for k in keys):
            found.append(label)

    # 중복 제거 + topn
    uniq: List[Dict[str, str]] = []
    seen = set()
    for x in found:
        if x in seen:
            continue
        seen.add(x)
        uniq.append({"keyword": x})
        if len(uniq) >= topn:
            break

    return uniq

# =======================
# Schemas
# =======================
class KeywordItem(BaseModel):
    keyword: str

class ReviewSummaryRes(BaseModel):
    productId: int
    positive: List[KeywordItem]
    negative: List[KeywordItem]
    totalCount: int
    analyzedAt: Optional[datetime] = None

# =======================
# SQL
# =======================
SUMMARY_SQL = """
SELECT
  product_id,
  pos_keyword_num,
  neg_keyword_num
FROM public.product_review_summary
WHERE product_id = %(product_id)s
LIMIT 1
"""

FEATURE_SQL = """
SELECT
  product_id,
  feature_scores
FROM public.product_review_keywords
WHERE product_id = %(product_id)s
LIMIT 1
"""

AI_SUMMARY_SQL = """
SELECT
  summary,
  updated_at
FROM public.ai_review_summary
WHERE product_id = %(product_id)s
ORDER BY updated_at DESC
LIMIT 1
"""

# =======================
# Route
# =======================
@router.get(
    "/{product_id}/review-keywords",
    response_model=ReviewSummaryRes,
    summary="상품별 긍정/부정 키워드 요약 API",
    description="상품별로 긍정, 부정 키워드와 총 리뷰수를 반환합니다.\n\n"
)
def get_review_summary(product_id: int, topn: int = 5):
    if topn < 1 or topn > 20:
        raise HTTPException(status_code=400, detail="topn must be between 1 and 20")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(SUMMARY_SQL, {"product_id": product_id})
                srow = cur.fetchone()

                cur.execute(FEATURE_SQL, {"product_id": product_id})
                frow = cur.fetchone()

                cur.execute(AI_SUMMARY_SQL, {"product_id": product_id})
                arow = cur.fetchone()

        if not srow and not frow and not arow:
            raise HTTPException(status_code=404, detail="summary not found")

        # totalCount
        total_count = 0
        if srow:
            total_count = int(srow.get("pos_keyword_num") or 0) + int(srow.get("neg_keyword_num") or 0)

        # positive
        positive: List[Dict[str, str]] = []
        if frow:
            fs = _coerce_json(frow.get("feature_scores"))
            if isinstance(fs, dict):
                positive = pack_feature_scores(fs, topn=topn)
        positive = beautify_positive_list(positive)

        # negative (ai summary -> short labels)
        negative: List[Dict[str, str]] = []
        analyzed_at: Optional[datetime] = None
        if arow and arow.get("summary"):
            negative = ai_summary_to_short_negatives(str(arow["summary"]), topn=topn)
        if arow and arow.get("updated_at"):
            analyzed_at = arow.get("updated_at")

        return {
            "productId": product_id,
            "positive": positive,
            "negative": negative,
            "totalCount": total_count,
            "analyzedAt": analyzed_at,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")
