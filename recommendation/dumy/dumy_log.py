# seed_dummy_events.py
import os
import random
from datetime import datetime, timedelta

import psycopg2
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

EVENTS = [
    ("CART_ADD", 0.70),
    ("CART_REMOVE", 0.20),
    ("CANCELED", 0.10),
]

def pick_event_type():
    r = random.random()
    acc = 0.0
    for name, p in EVENTS:
        acc += p
        if r <= acc:
            return name
    return EVENTS[-1][0]

def main():
    random.seed(42)

    conn = psycopg2.connect(build_dsn())
    cur = conn.cursor()

    cur.execute("""
        SELECT id, external_product_id
        FROM product
        WHERE external_product_id NOT LIKE 'INTERNAL-%'
        ORDER BY id;
    """)
    products = cur.fetchall()
    if not products:
        raise RuntimeError("No products found in product table.")

    NUM_MEMBERS = 200
    EVENTS_PER_MEMBER = 120
    DAYS_RANGE = 60

    rows = []
    now = datetime.now()

    for member_id in range(1, NUM_MEMBERS + 1):
        preferred = random.sample(products, k=min(50, len(products)))

        for _ in range(EVENTS_PER_MEMBER):
            event_type = pick_event_type()

            # CART_ADD는 더 선호 상품 위주로
            if event_type == "CART_ADD":
                pid, ext = random.choice(preferred[: min(15, len(preferred))])
                qty = 1
            else:
                # 제거/취소는 좀 더 넓게
                pid, ext = random.choice(preferred if random.random() < 0.7 else products)
                qty = 1

            occurred_at = now - timedelta(days=random.randint(0, DAYS_RANGE), hours=random.randint(0, 23))
            rows.append((member_id, pid, ext, event_type, qty, occurred_at))

    cur.executemany("""
        INSERT INTO rec_event_log (member_id, product_id, external_product_id, event_type, quantity, occurred_at)
        VALUES (%s, %s, %s, %s::rec_event_type, %s, %s);
    """, rows)

    conn.commit()
    cur.close()
    conn.close()
    print(f"[DONE] inserted {len(rows)} rows into rec_event_log")

if __name__ == "__main__":
    main()
