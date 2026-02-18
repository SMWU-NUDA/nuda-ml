import os
import argparse
import random
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Set, Optional

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

def fetch_users(cur, limit: int) -> List[int]:
    cur.execute("SELECT user_id FROM rec_user_pref ORDER BY user_id LIMIT %s", (limit,))
    return [int(r[0]) for r in cur.fetchall()]

def fetch_products(cur, limit: int) -> List[str]:
    cur.execute(
        """
        SELECT product_id
        FROM rec_product_feature
        WHERE product_id IS NOT NULL AND btrim(product_id) <> ''
        ORDER BY product_id
        LIMIT %s
        """,
        (limit,)
    )
    return [r[0] for r in cur.fetchall()]

def fetch_pos_pairs(cur, days: int) -> List[Tuple[int, str]]:
    cur.execute(
        """
        SELECT DISTINCT user_id, product_id
        FROM rec_event_log
        WHERE event_type='purchase'
          AND created_at >= now() - (%s || ' days')::interval
        """,
        (days,)
    )
    return [(int(r[0]), r[1]) for r in cur.fetchall()]

def fetch_neg_cart_pairs(cur, days: int) -> List[Tuple[int, str]]:
    cur.execute(
        """
        WITH pos AS (
          SELECT DISTINCT user_id, product_id
          FROM rec_event_log
          WHERE event_type='purchase'
            AND created_at >= now() - (%s || ' days')::interval
        ),
        cart AS (
          SELECT DISTINCT user_id, product_id
          FROM rec_event_log
          WHERE event_type='cart'
            AND created_at >= now() - (%s || ' days')::interval
        )
        SELECT c.user_id, c.product_id
        FROM cart c
        LEFT JOIN pos p
          ON p.user_id=c.user_id AND p.product_id=c.product_id
        WHERE p.user_id IS NULL
        """,
        (days, days)
    )
    return [(int(r[0]), r[1]) for r in cur.fetchall()]

def fetch_user_pref_map(cur) -> Dict[int, Dict[str, int]]:
    cur.execute(
        """
        SELECT user_id, sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level
        FROM rec_user_pref
        """
    )
    mp = {}
    for r in cur.fetchall():
        mp[int(r[0])] = {
            "u_sensitivity": int(r[1]),
            "u_scent": int(r[2]),
            "u_absorbency": int(r[3]),
            "u_adhesion": int(r[4]),
            "u_safety": int(r[5]),
        }
    return mp

def fetch_product_feat_map(cur) -> Dict[str, Dict[str, int]]:
    cur.execute(
        """
        SELECT
          product_id,
          sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level
        FROM rec_product_feature
        """
    )
    mp = {}
    for r in cur.fetchall():
        pid = r[0]
        mp[pid] = {
            "p_sensitivity": int(r[1]),
            "p_scent": int(r[2]),
            "p_absorbency": int(r[3]),
            "p_adhesion": int(r[4]),
            "p_safety": int(r[5]),
        }
    return mp

def fetch_log_features(cur, user_id: int, product_id: str) -> Tuple[int,int]:
    cur.execute(
        """
        SELECT
          COUNT(*) FILTER (WHERE event_type='cart' AND created_at >= now() - interval '30 days') AS cart_30d,
          COUNT(*) FILTER (WHERE event_type='purchase' AND created_at >= now() - interval '180 days') AS pur_180d
        FROM rec_event_log
        WHERE user_id=%s AND product_id=%s
        """,
        (user_id, product_id)
    )
    r = cur.fetchone()
    return int(r[0] or 0), int(r[1] or 0)

def build_risky_ingredient_list(cur) -> List[Tuple[str, int, str]]:
    cur.execute(
        """
        SELECT DISTINCT
          n.normalized_name,
          h.risk_score,
          h.risk_level
        FROM msds_hazard h
        JOIN ingredient_msds_mapping m ON m.chosen_candidate_id = h.chosen_candidate_id
        JOIN ingredient_normalized n ON n.id = m.ingredient_normalized_id
        WHERE jsonb_array_length(h.h_codes) > 0
          AND m.match_status='matched'
        """
    )
    return [(r[0], int(r[1]), r[2]) for r in cur.fetchall()]

def compute_hazard_features_by_ilike(cur, product_id: str, risky_list: List[Tuple[str,int,str]]) -> Tuple[int,int,int]:
    cur.execute(
        """
        SELECT material_name, sub_material
        FROM product_ingredient
        WHERE product_id=%s
        """,
        (product_id,)
    )
    rows = cur.fetchall()
    blob = " ".join([(r[0] or "") + " " + (r[1] or "") for r in rows])

    any_cnt, high_cnt, max_score = 0, 0, 0
    for name, score, level in risky_list:
        if name and name in blob:
            any_cnt += 1
            max_score = max(max_score, score)
            if level == "high":
                high_cnt += 1
    return any_cnt, high_cnt, max_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--user_limit", type=int, default=30000)
    ap.add_argument("--product_pool", type=int, default=30000)
    ap.add_argument("--pos_days", type=int, default=180)
    ap.add_argument("--neg_days", type=int, default=180)
    ap.add_argument("--neg_random_per_user", type=int, default=30)
    ap.add_argument("--batch", type=int, default=3000)
    ap.add_argument("--max_rows", type=int, default=2000000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    dsn = build_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    as_of = datetime.now(timezone.utc)

    try:
        with conn.cursor() as cur:
            if args.truncate:
                cur.execute("TRUNCATE TABLE rec_train_pair;")
                conn.commit()

            users = fetch_users(cur, args.user_limit)
            products = fetch_products(cur, args.product_pool)
            if not users:
                raise RuntimeError("rec_user_pref 유저 없음. seed_user_prefs.py 먼저")
            if not products:
                raise RuntimeError("rec_product_feature 상품 없음. seed_product_features.py 먼저")

            user_pref = fetch_user_pref_map(cur)
            prod_feat = fetch_product_feat_map(cur)

            pos = fetch_pos_pairs(cur, args.pos_days)
            neg_cart = fetch_neg_cart_pairs(cur, args.neg_days)

            pos_set: Set[Tuple[int,str]] = set(pos)
            neg_cart_set: Set[Tuple[int,str]] = set(neg_cart)

            risky_list = build_risky_ingredient_list(cur)

            rows = []
            inserted = 0

            def make_row(uid: int, pid: str, label: int) -> Optional[tuple]:
                up = user_pref.get(uid)
                pf = prod_feat.get(pid)
                if not up or not pf:
                    return None

                # diff 피쳐 (유저 선호와 상품 특성의 mismatch 정도)
                d_sens = abs(up["u_sensitivity"] - pf["p_sensitivity"])
                d_scent = abs(up["u_scent"] - pf["p_scent"])
                d_abs = abs(up["u_absorbency"] - pf["p_absorbency"])
                d_adh = abs(up["u_adhesion"] - pf["p_adhesion"])
                d_safe = abs(up["u_safety"] - pf["p_safety"])

                cart_30d, pur_180d = fetch_log_features(cur, uid, pid)
                any_cnt, high_cnt, max_score = compute_hazard_features_by_ilike(cur, pid, risky_list)

                return (
                    uid, pid, label, as_of,
                    up["u_sensitivity"], up["u_scent"], up["u_absorbency"], up["u_adhesion"], up["u_safety"],
                    any_cnt, high_cnt, max_score,
                    cart_30d, pur_180d,
                    pf["p_sensitivity"], pf["p_scent"], pf["p_absorbency"], pf["p_adhesion"], pf["p_safety"],
                    d_sens, d_scent, d_abs, d_adh, d_safe
                )

            def flush():
                nonlocal rows, inserted
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO rec_train_pair (
                      user_id, product_id, label, as_of,
                      sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level,
                      hazard_any_cnt, hazard_high_cnt, hazard_max_score,
                      cart_cnt_30d, purchase_cnt_180d,
                      p_sensitivity_level, p_scent_level, p_absorbency_level, p_adhesion_level, p_safety_level,
                      diff_sensitivity, diff_scent, diff_absorbency, diff_adhesion, diff_safety
                    ) VALUES %s
                    ON CONFLICT (user_id, product_id, as_of) DO NOTHING
                    """,
                    rows,
                    page_size=min(len(rows), args.batch)
                )
                inserted += len(rows)
                rows = []

            # 1) pos 먼저
            for uid, pid in pos:
                r = make_row(uid, pid, 1)
                if r:
                    rows.append(r)
                if len(rows) >= args.batch:
                    flush()
                    conn.commit()
                if inserted >= args.max_rows:
                    break

            if rows and inserted < args.max_rows:
                flush(); conn.commit()

            # 2) cart-only neg
            for uid, pid in neg_cart:
                if inserted >= args.max_rows:
                    break
                if (uid, pid) in pos_set:
                    continue
                r = make_row(uid, pid, 0)
                if r:
                    rows.append(r)
                if len(rows) >= args.batch:
                    flush(); conn.commit()

            if rows and inserted < args.max_rows:
                flush(); conn.commit()

            # 3) 랜덤 neg 폭발 생성
            for uid in users:
                if inserted >= args.max_rows:
                    break
                made = 0
                tries = 0
                while made < args.neg_random_per_user and tries < args.neg_random_per_user * 10:
                    tries += 1
                    pid = random.choice(products)
                    if (uid, pid) in pos_set:
                        continue
                    if (uid, pid) in neg_cart_set:
                        continue
                    r = make_row(uid, pid, 0)
                    if not r:
                        continue
                    rows.append(r)
                    made += 1
                    if len(rows) >= args.batch:
                        flush(); conn.commit()
                    if inserted >= args.max_rows:
                        break

            if rows and inserted < args.max_rows:
                flush(); conn.commit()

        print(f"[DONE] build_trainset_v2 inserted~={inserted} as_of={as_of.isoformat()}")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
