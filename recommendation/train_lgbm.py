import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

import lightgbm as lgb
from sklearn.model_selection import train_test_split

load_dotenv()

FEATURE_COLS = [
    "p_sensitivity_level", "p_scent_level", "p_absorbency_level", "p_adhesion_level", "p_safety_level",
    "d_sensitivity", "d_scent", "d_absorbency", "d_adhesion", "d_safety"
]

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
    conn = psycopg2.connect(build_dsn())

    sql = """
        SELECT
        user_id::bigint AS member_id,
        label::int      AS label,

        COALESCE(p_sensitivity_level, 3) AS p_sensitivity_level,
        COALESCE(p_scent_level, 3)       AS p_scent_level,
        COALESCE(p_absorbency_level, 3)  AS p_absorbency_level,
        COALESCE(p_adhesion_level, 3)    AS p_adhesion_level,
        COALESCE(p_safety_level, 3)      AS p_safety_level,

        COALESCE(d_sensitivity, 2)       AS d_sensitivity,
        COALESCE(d_scent, 2)             AS d_scent,
        COALESCE(d_absorbency, 2)        AS d_absorbency,
        COALESCE(d_adhesion, 2)          AS d_adhesion,
        COALESCE(d_safety, 2)            AS d_safety

        FROM rec_train_pair
        WHERE label IS NOT NULL;
        """


    df = pd.read_sql(sql, conn)
    conn.close()

    # ✅ group은 user_id별 연속 구간이어야 해서 정렬 필수
    df = df.sort_values("member_id").reset_index(drop=True)

    y = df["label"].astype(int)
    X = df[FEATURE_COLS]
    qid = df["member_id"].astype(int)

    # ✅ member 단위 split (같은 member가 train/val에 섞이면 랭킹 평가가 망가짐)
    members = qid.unique()
    train_members, val_members = train_test_split(members, test_size=0.2, random_state=42)

    train_mask = qid.isin(train_members)
    val_mask = qid.isin(val_members)

    # ✅ group sizes 만들기: q_train의 member별 row 수 리스트
    # 주의: Dataset에 넣을 group은 "각 그룹의 샘플 개수"를 순서대로 준 리스트면 됨
    train_df = df[train_mask].copy().sort_values("member_id").reset_index(drop=True)
    val_df   = df[val_mask].copy().sort_values("member_id").reset_index(drop=True)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["label"].astype(int)
    q_train = train_df["member_id"].astype(int)

    X_val = val_df[FEATURE_COLS]
    y_val = val_df["label"].astype(int)
    q_val = val_df["member_id"].astype(int)

    group_train = train_df.groupby("member_id").size().tolist()
    group_val   = val_df.groupby("member_id").size().tolist()


    train_data = lgb.Dataset(X_train, label=y_train, group=group_train, free_raw_data=False)
    val_data   = lgb.Dataset(X_val,   label=y_val,   group=group_val,   free_raw_data=False)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 20],
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=3000,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)],
    )

    out_path = "rec_lgbm_rank.txt"
    model.save_model(out_path)
    print(f"[DONE] saved model -> {out_path} (best_iter={model.best_iteration})")
    print("X_train columns:", list(X_train.columns), "n=", X_train.shape[1])


if __name__ == "__main__":
    main()
