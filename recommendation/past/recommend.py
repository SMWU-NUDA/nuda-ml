import os
import argparse
import pandas as pd
import psycopg2

from dotenv import load_dotenv
from lightgbm import Booster

load_dotenv()

def dsn():
    return (
        f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} dbname={os.getenv('DB_NAME')} "
        f"user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')}"
    )

FEATURES = [
    "sensitivity_level", "scent_level", "absorbency_level", "adhesion_level", "safety_level",
    "hazard_any_cnt", "hazard_high_cnt", "hazard_max_score",
    "cart_cnt_30d", "purchase_cnt_180d",
    "p_sensitivity_level", "p_scent_level", "p_absorbency_level", "p_adhesion_level", "p_safety_level",
    "diff_sensitivity", "diff_scent", "diff_absorbency", "diff_adhesion", "diff_safety",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_id", type=int, required=True)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--pool", type=int, default=5000, help="후보 상품 수 (너무 크면 느려짐)")
    ap.add_argument("--model", type=str, default="reco_lgbm_model_v2.txt")
    ap.add_argument("--mix_rule", type=float, default=0.8, help="cold-start에서 rule 비중 (0~1)")
    args = ap.parse_args()

    model = Booster(model_file=args.model)

    conn = psycopg2.connect(dsn())

    # 1) 유저 pref
    user_sql = """
    SELECT user_id, sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level
    FROM rec_user_pref
    WHERE user_id=%s
    """
    u = pd.read_sql(user_sql, conn, params=(args.user_id,))
    if u.empty:
        raise RuntimeError("user_id 없음. seed_user_prefs.py에서 만든 user_id로 테스트해줘")

    urow = u.iloc[0].to_dict()

    # 2) 후보 상품 pool: diff 작은 순으로
    prod_sql = """
    SELECT
      product_id,
      sensitivity_level AS p_sensitivity_level,
      scent_level AS p_scent_level,
      absorbency_level AS p_absorbency_level,
      adhesion_level AS p_adhesion_level,
      safety_level AS p_safety_level
    FROM rec_product_feature
    ORDER BY
      ABS(sensitivity_level - %s)
    + ABS(scent_level - %s)
    + ABS(absorbency_level - %s)
    + ABS(adhesion_level - %s)
    + ABS(safety_level - %s)
    LIMIT %s
    """
    p = pd.read_sql(
        prod_sql,
        conn,
        params=(
            int(urow["sensitivity_level"]),
            int(urow["scent_level"]),
            int(urow["absorbency_level"]),
            int(urow["adhesion_level"]),
            int(urow["safety_level"]),
            args.pool,
        ),
    )

    conn.close()

    # 3) 유저 x 상품 조합 피쳐 만들기
    df = p.copy()
    df["sensitivity_level"] = int(urow["sensitivity_level"])
    df["scent_level"] = int(urow["scent_level"])
    df["absorbency_level"] = int(urow["absorbency_level"])
    df["adhesion_level"] = int(urow["adhesion_level"])
    df["safety_level"] = int(urow["safety_level"])

    # cold-start: 로그/위험성은 0
    df["hazard_any_cnt"] = 0
    df["hazard_high_cnt"] = 0
    df["hazard_max_score"] = 0
    df["cart_cnt_30d"] = 0
    df["purchase_cnt_180d"] = 0

    df["diff_sensitivity"] = (df["sensitivity_level"] - df["p_sensitivity_level"]).abs()
    df["diff_scent"] = (df["scent_level"] - df["p_scent_level"]).abs()
    df["diff_absorbency"] = (df["absorbency_level"] - df["p_absorbency_level"]).abs()
    df["diff_adhesion"] = (df["adhesion_level"] - df["p_adhesion_level"]).abs()
    df["diff_safety"] = (df["safety_level"] - df["p_safety_level"]).abs()

    X = df[FEATURES].fillna(0)

    # 4) 모델 점수
    df["score"] = model.predict(X)

    # 5) 정규화 + rule 혼합 (pool 내에서만 정규화)
    ms = df["score"]
    df["model_score_norm"] = (ms - ms.min()) / (ms.max() - ms.min() + 1e-12)

    diff_sum = (
        df["diff_sensitivity"] + df["diff_scent"] + df["diff_absorbency"] +
        df["diff_adhesion"] + df["diff_safety"]
    )
    df["rule_score"] = 1 / (1 + diff_sum)

    w_rule = float(args.mix_rule)
    w_model = 1.0 - w_rule
    df["final_score"] = w_model * df["model_score_norm"] + w_rule * df["rule_score"]

    # 6) 결과 출력
    out = df.sort_values("final_score", ascending=False).head(args.topn)[
        ["product_id", "final_score", "model_score_norm", "rule_score",
         "p_sensitivity_level", "p_absorbency_level", "p_safety_level",
         "diff_sensitivity", "diff_absorbency", "diff_safety"]
    ]
    print(out.to_string(index=False))

    # (디버그) 모델 점수 분포
    print("\n[DEBUG] raw score min/max:", df["score"].min(), df["score"].max())
    print("[DEBUG] raw score std:", df["score"].std())

if __name__ == "__main__":
    main()
