import os
import argparse
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import lightgbm as lgb

load_dotenv()

def must_env(k: str) -> str:
    v = os.getenv(k)
    if not v:
        raise RuntimeError(f"Missing env var: {k}")
    return v

def dsn():
    return (
        f"host={must_env('DB_HOST')} port={must_env('DB_PORT')} "
        f"dbname={must_env('DB_NAME')} user={must_env('DB_USER')} password={must_env('DB_PASSWORD')}"
    )

def fetch_df(sql: str) -> pd.DataFrame:
    with psycopg2.connect(dsn()) as conn:
        return pd.read_sql(sql, conn)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="reco_lgbm_model.txt")
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    booster = lgb.Booster(model_file=args.model)

    # user prefs
    u = fetch_df(f"""
      SELECT
        user_id,
        sensitivity_level,
        scent_level,
        absorbency_level,
        adhesion_level,
        safety_level
      FROM rec_user_pref_v
      WHERE user_id = {args.user_id};
    """)
    if u.empty:
        raise RuntimeError("No prefs found for this user_id in rec_user_pref_v")

    # candidate items: 일단 전체 product (나중에 카테고리 필터/품절 필터 등 추가)
    items = fetch_df("""
      SELECT
        product_id,
        name,
        brand_name,
        category_code,
        hazard_any_cnt,
        hazard_high_cnt,
        hazard_max_score
      FROM rec_user_item_v;
    """)

    # exclude already purchased by user
    purchased = fetch_df(f"""
      SELECT DISTINCT product_id
      FROM user_event
      WHERE user_id = {args.user_id} AND event_type = 'purchase';
    """)
    if not purchased.empty:
        items = items[~items["product_id"].isin(purchased["product_id"])]

    # feature matrix
    for col in ["hazard_any_cnt","hazard_high_cnt","hazard_max_score"]:
        items[col] = items[col].fillna(0)

    for col in ["sensitivity_level","scent_level","absorbency_level","adhesion_level","safety_level"]:
        items[col] = int(u.iloc[0][col])

    X = items[[
        "sensitivity_level","scent_level","absorbency_level","adhesion_level","safety_level",
        "hazard_any_cnt","hazard_high_cnt","hazard_max_score"
    ]]

    scores = booster.predict(X)
    items["score"] = scores

    out = items.sort_values("score", ascending=False).head(args.topn)
    print(out[["product_id","name","brand_name","category_code","score","hazard_max_score","hazard_high_cnt"]].to_string(index=False))

if __name__ == "__main__":
    main()
