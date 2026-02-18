# review_generate_deploy.py
import os, json, re
from typing import Optional, Dict, Any, List

import psycopg
import requests
from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

# =========================
# Prompt
# =========================
PROMPT_TEMPLATE = """너는 상품 리뷰 데이터를 바탕으로 '주목할만한 특징' 3문장을 작성합니다.

[규칙]
- 문장은 최대 3문장으로 구성합니다.
- 리뷰를 분석한 결과를 "상품 설명 요약"처럼 작성합니다.
- 기호는 사용하지 않습니다.
- 무조건 "~입니다.", "~습니다."로 마무리합니다.
- 입력에 있는 근거만 사용합니다. (성분/의학/안전성 등 추정 금지)
- 1문장: 전체 톤(긍정/부정 비율을 반영) + 핵심 특징 1개
- 2문장: 가장 강한 feature 기반으로 구체화
- 3문장: 주의 포인트 1개(근거 기반). 근거가 약하면 "개인차/근거 부족"으로 처리
- '사람들', '사용자들은' 같은 표현은 최소화합니다.
- 한 문장에는 하나의 핵심만 담습니다.
- 과장된 표현이나 광고 문구는 사용하지 않습니다.

[입력 JSON]
{input_json}

이제 3문장만 출력합니다.
"""

# =========================
# DB
# =========================
def db_conn():
    conn = psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute("SELECT current_database(), inet_server_addr(), inet_server_port(), current_user;")
        print("[DB-CONNECT]", cur.fetchone())
    return conn

# =========================
# Ollama
# =========================
def call_ollama_text(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "num_predict": 220
            }
        },
        timeout=180,
    )
    r.raise_for_status()
    text = (r.json().get("response") or "").strip()
    return " ".join(text.split())

# =========================
# Target selection (last_review_at-based)
# =========================
def fetch_target_external_ids_by_lastreview(
    conn,
    limit: Optional[int] = None,
    min_reviews: int = 10,
) -> List[str]:
    """
    - v_product_review_stats(review_count, last_review_at) 기준
    - ai_review_summary가 없거나, last_review_at > ai_review_summary.updated_at 이면 재생성 대상
    - META/스키마 변경 없이 증분 갱신 가능
    """
    sql = """
        SELECT
          p.external_product_id
        FROM product p
        JOIN v_product_review_stats v
          ON v.product_id = p.id
        LEFT JOIN ai_review_summary a
          ON a.external_product_id = p.external_product_id
        WHERE p.external_product_id IS NOT NULL
          AND v.review_count >= %s
          AND (
            a.external_product_id IS NULL
            OR (v.last_review_at IS NOT NULL AND v.last_review_at > a.updated_at)
          )
        ORDER BY v.last_review_at DESC NULLS LAST, v.review_count DESC, p.id DESC
    """
    params: List[Any] = [min_reviews]
    if limit:
        sql += " LIMIT %s"
        params.append(limit)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [r[0] for r in cur.fetchall()]

# =========================
# Payload (uses summary/keywords tables)
# =========================
def fetch_payload(conn, ext_id: str) -> Optional[Dict[str, Any]]:
    """
    NOTE:
    - product_review_summary / product_review_keywords 가 먼저 채워져 있어야 함
    - review_count는 v_product_review_stats 기준으로 가져옴
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              p.id AS product_id,
              s.external_product_id,
              v.review_count,
              s.pos_ratio,
              s.neg_ratio,
              k.top_keywords,
              k.aspect_keywords,
              k.feature_keywords,
              k.feature_scores
            FROM product_review_summary s
            JOIN product_review_keywords k
              ON s.external_product_id = k.external_product_id
            JOIN product p
              ON p.id = s.product_id
            JOIN v_product_review_stats v
              ON v.product_id = p.id
            WHERE s.external_product_id = %s
            """,
            (ext_id,),
        )
        row = cur.fetchone()

    if not row:
        return None

    (product_id, external_product_id, review_count,
     pos_ratio, neg_ratio, top_keywords, aspect_keywords,
     feature_keywords, feature_scores) = row

    def to_obj(x):
        if x is None:
            return {}
        if isinstance(x, (dict, list)):
            return x
        if isinstance(x, str):
            return json.loads(x)
        return x

    return {
        "product_id": int(product_id),
        "external_product_id": external_product_id,
        "review_count": int(review_count) if review_count is not None else 0,
        "pos_ratio": float(pos_ratio) if pos_ratio is not None else None,
        "neg_ratio": float(neg_ratio) if neg_ratio is not None else None,
        "top_keywords": to_obj(top_keywords),
        "aspect_keywords": to_obj(aspect_keywords),
        "feature_keywords": to_obj(feature_keywords),
        "feature_scores": to_obj(feature_scores),
        "product_type": "생리대",
    }

# =========================
# Upsert (no META)
# =========================
def upsert_ai_review_summary(conn, payload: Dict[str, Any], summary_text: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO ai_review_summary (product_id, external_product_id, summary, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (external_product_id) DO UPDATE SET
              product_id = EXCLUDED.product_id,
              summary = EXCLUDED.summary,
              updated_at = now()
            """,
            (payload["product_id"], payload["external_product_id"], (summary_text or "").strip()),
        )
    conn.commit()

# =========================
# Runner
# =========================
def run_incremental(
    limit: Optional[int] = None,
    min_reviews: int = 10,
):
    with db_conn() as conn:
        ext_ids = fetch_target_external_ids_by_lastreview(
            conn, limit=limit, min_reviews=min_reviews
        )
        print("to update:", len(ext_ids))

        for ext_id in ext_ids:
            payload = fetch_payload(conn, ext_id)
            if not payload:
                print("[SKIP] payload missing for:", ext_id)
                continue

            prompt = PROMPT_TEMPLATE.format(
                input_json=json.dumps(payload, ensure_ascii=False)
            )
            summary_text = call_ollama_text(prompt)
            upsert_ai_review_summary(conn, payload, summary_text)
            print("updated:", ext_id, "review_count:", payload.get("review_count", 0))

if __name__ == "__main__":
    # 운영 추천값: min_reviews=10
    run_incremental(limit=940, min_reviews=10)
