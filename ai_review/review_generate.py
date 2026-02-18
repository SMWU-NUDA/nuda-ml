import os, json
import psycopg
import requests
from dotenv import load_dotenv

env_file = os.getenv("ENV_FILE", ".env.local")
load_dotenv(env_file)

# ===== 여기만 바꾸면 됨 =====
MODE = "PROD"  
# ===========================

if MODE == "PROD":
    load_dotenv(".env.prod")
else:
    load_dotenv(".env.local")


from typing import Optional, Dict, Any


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

PROMPT_TEMPLATE = """
너는 상품 리뷰 데이터를 바탕으로 '주목할만한 특징' 3문장을 작성합니다. 
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

def ensure_ai_review_summary_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ai_review_summary (
              external_product_id TEXT PRIMARY KEY,
              category_code TEXT,
              summary TEXT,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
    conn.commit()


def call_ollama_text(prompt: str) -> str:
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=180,
    )
    r.raise_for_status()
    text = (r.json().get("response") or "").strip()
    # 혹시 줄바꿈 섞이면 한 줄로
    return " ".join(text.split())

def fetch_stale_product_ids(conn, limit=None):
    sql = """
        SELECT s.external_product_id
        FROM product_review_summary s
        LEFT JOIN ai_review_summary a
          ON a.external_product_id = s.external_product_id
        WHERE a.external_product_id IS NULL
           OR s.updated_at > a.updated_at
        ORDER BY s.updated_at DESC
    """
    if limit:
        sql += " LIMIT %s"
        params = (limit,)
    else:
        params = None

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [r[0] for r in cur.fetchall()]

def fetch_payload(conn, pid: str) -> Optional[Dict[str, Any]]:
    # summary + keywords 조인 (너가 이미 만들어둔 feature 포함 컬럼 기준)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
              s.external_product_id,
              s.category_code,
              s.review_count,
              s.pos_ratio,
              s.neg_ratio,
              s.summary_3lines,
              s.top_aspects,
              k.feature_keywords,
              k.feature_scores
            FROM product_review_summary s
            JOIN product_review_keywords k
              ON s.external_product_id = k.external_product_id
            WHERE s.external_product_id = %s
        """, (pid,))
        row = cur.fetchone()

    if not row:
        return None

    (pid, category_code, review_count, pos_ratio, neg_ratio,
     summary_3lines, top_aspects, feature_keywords, feature_scores) = row

    def to_obj(x):
        if x is None:
            return None
        if isinstance(x, (dict, list)):
            return x
        if isinstance(x, str):
            return json.loads(x)
        return x

    return {
        "external_product_id": pid,
        "category_code": category_code,
        "review_count": int(review_count),
        "pos_ratio": float(pos_ratio) if pos_ratio is not None else None,
        "neg_ratio": float(neg_ratio) if neg_ratio is not None else None,
        "summary_3lines": summary_3lines,
        "top_aspects": to_obj(top_aspects) or {},
        "feature_keywords": to_obj(feature_keywords) or {},
        "feature_scores": to_obj(feature_scores) or {},
    }

def upsert_ai_review_summary(conn, pid: str, category_code: str, summary_text: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO ai_review_summary (external_product_id, category_code, summary, updated_at)
            VALUES (%s, %s, %s, now())
            ON CONFLICT (external_product_id) DO UPDATE SET
              category_code = EXCLUDED.category_code,
              summary = EXCLUDED.summary,
              updated_at = now()
        """, (pid, category_code, summary_text))
    conn.commit()

def run_incremental(limit=None):
    with psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    ) as conn:
        ensure_ai_review_summary_table(conn)
    
        pids = fetch_stale_product_ids(conn, limit=limit)
        print("to update:", len(pids))

        for pid in pids:
            payload = fetch_payload(conn, pid)
            if not payload:
                continue
            payload["product_type"] = "생리대"
            
            prompt = PROMPT_TEMPLATE.format(
                product_type=payload.get("product_type", "생리대"),
                input_json=json.dumps(payload, ensure_ascii=False)
                )

            summary_text = call_ollama_text(prompt)

            upsert_ai_review_summary(conn, pid, payload["category_code"], summary_text)
            print("updated:", pid)

if __name__ == "__main__":
    run_incremental(limit=10)  # 테스트는 50개만
