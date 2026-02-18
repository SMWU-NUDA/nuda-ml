import os
import random
import argparse

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

def must_env(name: str) -> str:
    v = os.getenv(name)
    if not v or v.strip() == "":
        raise RuntimeError(f"Missing env var: {name}")
    return v.strip()

def build_dsn() -> str:
    return (
        f"host={must_env('DB_HOST')} port={must_env('DB_PORT')} dbname={must_env('DB_NAME')} "
        f"user={must_env('DB_USER')} password={must_env('DB_PASSWORD')}"
    )

def clamp(x: int, lo=1, hi=5) -> int:
    return max(lo, min(hi, x))

def biased_level(base: int, bias: int) -> int:
    return clamp(base + bias)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--batch", type=int, default=5000)
    args = ap.parse_args()

    random.seed(args.seed)
    dsn = build_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            if args.reset:
                cur.execute("TRUNCATE TABLE rec_product_feature;")
                conn.commit()

            cur.execute(
                """
                SELECT product_id, category_code, brand_name
                FROM product
                WHERE product_id IS NOT NULL AND btrim(product_id) <> ''
                ORDER BY product_id
                LIMIT %s
                """,
                (args.limit,)
            )
            products = cur.fetchall()

            rows = []
            for pid, cat, brand in products:
                # 기본 랜덤 (1~5)
                base_sens = random.randint(1, 5)
                base_scent = random.randint(1, 5)
                base_abs = random.randint(1, 5)
                base_adh = random.randint(1, 5)
                base_safe = random.randint(1, 5)

                # 카테고리 편향(예시)
                # - 오버나이트/대형은 흡수력 높은 경향
                # - 라이너는 흡수력 낮고 민감도 높은 경향
                bias_abs = 0
                bias_sens = 0

                if cat:
                    c = str(cat).upper()
                    if "OVER" in c or "NIGHT" in c:
                        bias_abs += 1
                    if "L" in c and "LINER" in c:
                        bias_abs -= 1
                        bias_sens += 1

                sens = biased_level(base_sens, bias_sens)
                scent = base_scent
                abs_ = biased_level(base_abs, bias_abs)
                adh = base_adh
                safe = base_safe

                rows.append((pid, cat, brand, sens, scent, abs_, adh, safe))

                if len(rows) >= args.batch:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO rec_product_feature (
                          product_id, category_code, brand_name,
                          sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level
                        )
                        VALUES %s
                        ON CONFLICT (product_id) DO UPDATE
                        SET category_code=EXCLUDED.category_code,
                            brand_name=EXCLUDED.brand_name,
                            sensitivity_level=EXCLUDED.sensitivity_level,
                            scent_level=EXCLUDED.scent_level,
                            absorbency_level=EXCLUDED.absorbency_level,
                            adhesion_level=EXCLUDED.adhesion_level,
                            safety_level=EXCLUDED.safety_level,
                            updated_at=now()
                        """,
                        rows,
                        page_size=len(rows)
                    )
                    conn.commit()
                    rows = []

            if rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO rec_product_feature (
                      product_id, category_code, brand_name,
                      sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level
                    )
                    VALUES %s
                    ON CONFLICT (product_id) DO UPDATE
                    SET category_code=EXCLUDED.category_code,
                        brand_name=EXCLUDED.brand_name,
                        sensitivity_level=EXCLUDED.sensitivity_level,
                        scent_level=EXCLUDED.scent_level,
                        absorbency_level=EXCLUDED.absorbency_level,
                        adhesion_level=EXCLUDED.adhesion_level,
                        safety_level=EXCLUDED.safety_level,
                        updated_at=now()
                    """,
                    rows,
                    page_size=len(rows)
                )
                conn.commit()

        print(f"[DONE] seeded rec_product_feature rows={len(products)}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
