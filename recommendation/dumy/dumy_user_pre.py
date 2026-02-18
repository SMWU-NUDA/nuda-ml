import os
import random
import argparse
from datetime import datetime, timezone

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--batch", type=int, default=10000)
    args = ap.parse_args()

    random.seed(args.seed)
    dsn = build_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            if args.reset:
                cur.execute("TRUNCATE TABLE rec_user_pref;")
                conn.commit()

            rows = []
            for uid in range(1, args.users + 1):
                # 1~5 (유저마다 다양하게)
                s = random.randint(1, 5)
                scent = random.randint(1, 5)
                ab = random.randint(1, 5)
                ad = random.randint(1, 5)
                safe = random.randint(1, 5)

                rows.append((uid, s, scent, ab, ad, safe))

                if len(rows) >= args.batch:
                    psycopg2.extras.execute_values(
                        cur,
                        """
                        INSERT INTO rec_user_pref
                          (user_id, sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level)
                        VALUES %s
                        ON CONFLICT (user_id) DO UPDATE
                        SET sensitivity_level=EXCLUDED.sensitivity_level,
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
                    INSERT INTO rec_user_pref
                      (user_id, sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level)
                    VALUES %s
                    ON CONFLICT (user_id) DO UPDATE
                    SET sensitivity_level=EXCLUDED.sensitivity_level,
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

        print(f"[DONE] seeded rec_user_pref users={args.users}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
