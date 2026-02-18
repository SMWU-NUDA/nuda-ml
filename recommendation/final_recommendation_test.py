import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import lightgbm as lgb

load_dotenv()

def dsn():
    return (
        f"host={os.getenv('DB_HOST')} port={os.getenv('DB_PORT')} dbname={os.getenv('DB_NAME')} "
        f"user={os.getenv('DB_USER')} password={os.getenv('DB_PASSWORD')}"
    )

USER_ID = 1
MODEL_PATH = "rec_lgbm_rank.txt"   # ✅ 방금 학습한 모델로
TOPK = 20

FEATURE_COLS = [
    "p_sensitivity_level","p_scent_level","p_absorbency_level","p_adhesion_level","p_safety_level",
    "d_sensitivity","d_scent","d_absorbency","d_adhesion","d_safety"
]

def main():
    model = lgb.Booster(model_file=MODEL_PATH)

    conn = psycopg2.connect(dsn())

    # 1) 유저 선호 불러오기
    u = pd.read_sql(
        """
        SELECT user_id,
               sensitivity_level AS p_sensitivity_level,
               scent_level       AS p_scent_level,
               absorbency_level  AS p_absorbency_level,
               adhesion_level    AS p_adhesion_level,
               safety_level      AS p_safety_level
        FROM rec_user_pref
        WHERE user_id = %s
        """,
        conn,
        params=(USER_ID,)
    )
    if u.empty:
        raise RuntimeError(f"user_id={USER_ID} not found in rec_user_pref")

    urow = u.iloc[0].to_dict()

    # 2) 상품 후보
    p = pd.read_sql(
        """
        SELECT external_product_id,
               sensitivity_level, scent_level, absorbency_level, adhesion_level, safety_level
        FROM rec_product_feature
        """,
        conn
    )
    conn.close()

    # 3) p_* 붙이기
    for k, v in urow.items():
        if k != "user_id":
            p[k] = v

    # 4) d_* 계산
    p["d_sensitivity"] = (p["p_sensitivity_level"] - p["sensitivity_level"]).abs()
    p["d_scent"]       = (p["p_scent_level"] - p["scent_level"]).abs()
    p["d_absorbency"]  = (p["p_absorbency_level"] - p["absorbency_level"]).abs()
    p["d_adhesion"]    = (p["p_adhesion_level"] - p["adhesion_level"]).abs()
    p["d_safety"]      = (p["p_safety_level"] - p["safety_level"]).abs()

    # ✅ 모델 입력은 10개만
    scores = model.predict(p[FEATURE_COLS], num_iteration=model.best_iteration)
    p["score"] = scores

    out = p.sort_values("score", ascending=False).head(TOPK)[
        ["external_product_id","score",
         "sensitivity_level","scent_level","absorbency_level","adhesion_level","safety_level"]
    ]
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
