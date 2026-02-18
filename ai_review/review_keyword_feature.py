"""
fill_review_feature_score.py

목표:
- product_review_keywords.feature_scores (JSONB) + review(content) 기반으로
- review_feature_score(product_id, review_id, feature_key, score, matched, review_updated_at, computed_at)
  테이블을 초기 적재(또는 재계산)한다.

실행 예시:
  1) 전체 상품 최초 적재:
     python fill_review_feature_score.py

  2) 특정 상품만:
     python fill_review_feature_score.py --product-ids 123 456 789

  3) 이미 계산된 것 스킵(리뷰 update 없는 상품은 건너뜀):
     python fill_review_feature_score.py --only-if-stale

  4) 무조건 전부 재계산:
     python fill_review_feature_score.py --force
"""

import os
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv


# =========================
# ENV
# =========================
# 필요하면 --env-file로 바꿀 수 있게 해둠
def load_env(env_file: str):
    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv()  # 기본 .env


def get_conn():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        row_factory=dict_row,
        connect_timeout=10,
        autocommit=False,
    )


# =========================
# CONFIG
# =========================
FEATURE_KEYS = ["민감/자극", "향/냄새", "흡수/샘", "접착/고정"]


# =========================
# SCORING
# =========================
def normalize_text(s: str) -> str:
    return (s or "").strip()


def score_and_match(content: str, kw_dict: Dict[str, Any]) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    너가 쓰던 방식 그대로:
    - substring 매칭
    - 매칭된 phrase 중 weight 최대값을 score로
    - matched에는 best phrase/weight 저장
    """
    text = normalize_text(content)
    if not text or not isinstance(kw_dict, dict) or not kw_dict:
        return 0.0, None

    best_w = 0.0
    best_phrase = None

    for phrase, w in kw_dict.items():
        if not phrase:
            continue
        if phrase in text:
            try:
                fw = float(w)
            except Exception:
                continue
            if fw > best_w:
                best_w = fw
                best_phrase = phrase

    if best_w < 0.0:
        best_w = 0.0
    if best_w > 1.0:
        best_w = 1.0

    matched = None
    if best_phrase is not None:
        matched = {"phrase": best_phrase, "weight": float(best_w)}

    return float(best_w), matched


# =========================
# SQL
# =========================
SQL_PRODUCTS = """
SELECT product_id, feature_scores
FROM product_review_keywords
WHERE feature_scores IS NOT NULL
"""

SQL_PRODUCTS_BY_IDS = SQL_PRODUCTS + " AND product_id = ANY(%(product_ids)s)"

SQL_REVIEWS = """
SELECT id, content, updated_at
FROM review
WHERE product_id = %(product_id)s
ORDER BY id
"""

SQL_LAST_COMPUTED_REVIEW_TS = """
SELECT MAX(review_updated_at) AS last_review_updated_at
FROM review_feature_score
WHERE product_id = %(product_id)s
"""

SQL_UPSERT = """
INSERT INTO review_feature_score (
  product_id, review_id, feature_key, score, matched, review_updated_at, computed_at
)
VALUES (
  %(product_id)s, %(review_id)s, %(feature_key)s, %(score)s, %(matched)s::jsonb, %(review_updated_at)s, now()
)
ON CONFLICT (product_id, review_id, feature_key)
DO UPDATE SET
  score = EXCLUDED.score,
  matched = EXCLUDED.matched,
  review_updated_at = EXCLUDED.review_updated_at,
  computed_at = now()
"""


# =========================
# MAIN
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default="", help="예: .env.local / .env.prod")
    ap.add_argument("--product-ids", nargs="*", type=int, default=None, help="특정 product_id만 처리")
    ap.add_argument("--only-if-stale", action="store_true", help="이미 계산됐고 리뷰가 안 변했으면 skip")
    ap.add_argument("--force", action="store_true", help="무조건 전부 재계산(only-if-stale 무시)")
    ap.add_argument("--limit-products", type=int, default=9999999, help="안전장치")
    ap.add_argument("--limit-reviews-per-product", type=int, default=9999999, help="안전장치")
    ap.add_argument("--commit-every", type=int, default=2000, help="몇 row마다 commit할지(너무 크면 메모리/락 길어짐)")
    args = ap.parse_args()

    load_env(args.env_file)

    rows_upserted = 0
    products_processed = 0
    products_skipped = 0

    with get_conn() as conn:
        try:
            # 1) 대상 상품 + feature_scores 가져오기
            if args.product_ids:
                products = conn.execute(
                    SQL_PRODUCTS_BY_IDS + " ORDER BY product_id LIMIT %(limit)s",
                    {"product_ids": args.product_ids, "limit": args.limit_products},
                ).fetchall()
            else:
                products = conn.execute(
                    SQL_PRODUCTS + " ORDER BY product_id LIMIT %(limit)s",
                    {"limit": args.limit_products},
                ).fetchall()

            print(f"[INFO] products to process: {len(products)}")

            # 2) 상품별 처리
            for p in products:
                product_id = int(p["product_id"])
                feature_scores = p["feature_scores"]

                if not isinstance(feature_scores, dict):
                    products_skipped += 1
                    continue

                # 2-1) 리뷰 가져오기
                reviews = conn.execute(
                    SQL_REVIEWS + " LIMIT %(limit_reviews)s",
                    {"product_id": product_id, "limit_reviews": args.limit_reviews_per_product},
                ).fetchall()

                if not reviews:
                    products_skipped += 1
                    continue

                # 2-2) stale이면 skip (force면 무시)
                if args.only_if_stale and (not args.force):
                    last = conn.execute(SQL_LAST_COMPUTED_REVIEW_TS, {"product_id": product_id}).fetchone()
                    last_ts = last["last_review_updated_at"] if last else None

                    cur_max = None
                    for r in reviews:
                        ts = r.get("updated_at")
                        if ts and (cur_max is None or ts > cur_max):
                            cur_max = ts

                    if last_ts is not None and cur_max is not None and last_ts >= cur_max:
                        products_skipped += 1
                        continue

                # 2-3) feature_key별 dict 준비
                feature_dicts: Dict[str, Dict[str, Any]] = {}
                for fk in FEATURE_KEYS:
                    d = feature_scores.get(fk, {})
                    feature_dicts[fk] = d if isinstance(d, dict) else {}

                # 2-4) 리뷰 x feature_key upsert
                local_count = 0
                for r in reviews:
                    review_id = int(r["id"])
                    content = r.get("content") or ""
                    review_updated_at = r.get("updated_at")

                    for feature_key, kw_dict in feature_dicts.items():
                        score, matched = score_and_match(content, kw_dict)

                        conn.execute(
                            SQL_UPSERT,
                            {
                                "product_id": product_id,
                                "review_id": review_id,
                                "feature_key": feature_key,
                                "score": score,
                                "matched": json.dumps(matched) if matched is not None else None,
                                "review_updated_at": review_updated_at,
                            },
                        )
                        rows_upserted += 1
                        local_count += 1

                        # 주기적 commit
                        if args.commit_every > 0 and rows_upserted % args.commit_every == 0:
                            conn.commit()
                            print(f"[COMMIT] upserted={rows_upserted}")

                products_processed += 1
                print(f"[DONE] product_id={product_id} upserted_rows={local_count}")

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise

    print("========== SUMMARY ==========")
    print(f"productsProcessed: {products_processed}")
    print(f"productsSkipped:   {products_skipped}")
    print(f"rowsUpserted:      {rows_upserted}")


if __name__ == "__main__":
    main()
