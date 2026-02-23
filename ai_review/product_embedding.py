import os
import math
from typing import List, Tuple, Optional

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
BATCH_SIZE = int(os.getenv("EMB_BATCH_SIZE", "64"))   
TEXT_MAX = int(os.getenv("EMB_TEXT_MAX", "2000"))     


def build_text(name: str, content: Optional[str]) -> str:
    name = (name or "").strip()
    content = (content or "").strip()
    text = (name + "\n" + content).strip()
    if len(text) > TEXT_MAX:
        text = text[:TEXT_MAX]
    return text

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

def main():
    load_dotenv()

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    if dim != 768:
        raise RuntimeError(f"Unexpected embedding dim: {dim} (expected 768)")

    conn = get_conn()
    conn.autocommit = False

    select_sql = """
    SELECT p.id, p.name, p.content
    FROM product p
    LEFT JOIN ml_product_embedding e ON e.product_id = p.id
    WHERE p.external_product_id NOT LIKE 'INTERNAL-%'
      AND e.product_id IS NULL
    ORDER BY p.id
    """

    upsert_sql = """
    INSERT INTO ml_product_embedding(product_id, embedding, updated_at)
    VALUES (%s, %s, now())
    ON CONFLICT (product_id)
    DO UPDATE SET embedding = EXCLUDED.embedding, updated_at = now()
    """

    try:
        with conn.cursor() as cur:
            cur.execute(select_sql)
            rows = cur.fetchall()

        total = len(rows)
        print(f"[INFO] to_embed={total}")

        if total == 0:
            print("[INFO] Nothing to do.")
            return

        num_batches = math.ceil(total / BATCH_SIZE)

        for b in range(num_batches):
            start = b * BATCH_SIZE
            end = min((b + 1) * BATCH_SIZE, total)
            batch = rows[start:end]

            texts: List[str] = []
            pids: List[int] = []
            for r in batch:
                pids.append(r["id"])
                texts.append(build_text(r["name"], r["content"]))


            embs = model.encode(
                texts,
                batch_size=len(texts),
                show_progress_bar=False,
                normalize_embeddings=True,  
            )

            params: List[Tuple[int, list]] = [(pid, emb.tolist()) for pid, emb in zip(pids, embs)]

            with conn.cursor() as cur:
                cur.executemany(upsert_sql, params)

            conn.commit()
            print(f"[OK] batch {b+1}/{num_batches} inserted={len(params)} (product_id {pids[0]}..{pids[-1]})")

        print("[DONE] All embeddings inserted.")

    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    main()