# app/routers/keyword_recommend_router.py

from typing import Literal, List
import os
import hashlib

import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Query

from ..db import get_conn
from ..reco_model import RecoModel
from ..sql import EMB_CANDIDATE_IDS_SQL, CANDIDATE_BY_IDS_SQL

router = APIRouter(prefix="/ml/products", tags=["products"])

MODEL_PATH = os.getenv("RECO_MODEL_PATH", "./models/rec_lgbm_rank.txt")
TOPK_DEFAULT = int(os.getenv("RECO_TOPK", "20"))
CANDIDATES_PER_CAT = int(os.getenv("RECO_CANDIDATES_PER_CAT", "200"))

model = RecoModel(MODEL_PATH)

Keyword = Literal["default", "irritationLevel", "scent", "absorption", "adhesion"]


def d_to_score(d: pd.Series) -> pd.Series:
    return (5 - d).clip(lower=0)


def apply_filter_rerank(df: pd.DataFrame, keyword: Keyword) -> pd.DataFrame:
    if keyword == "default":
        df["final_score"] = df["model_score"]
        return df

    col_map = {
        "irritationLevel": "d_sensitivity",
        "scent": "d_scent",
        "absorption": "d_absorbency",
        "adhesion": "d_adhesion",
    }
    dcol = col_map.get(keyword)
    if not dcol or dcol not in df.columns:
        raise HTTPException(status_code=500, detail=f"missing column: {dcol}")

    df["filter_score"] = d_to_score(pd.to_numeric(df[dcol], errors="coerce").fillna(0))
    df["final_score"] = df["model_score"] + df["filter_score"] * 0.2
    return df


PREF_SQL = """
SELECT
  p_sensitivity_level,
  p_scent_level,
  p_absorbency_level,
  p_adhesion_level,
  p_safety_level
FROM rec_member_pref
WHERE member_id = %s
"""


def _to_float(x, default=3.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def pref_to_query_vec(pref: dict, dim: int = 768) -> List[float]:
    vals = np.array(
        [
            _to_float(pref.get("p_sensitivity_level", 3)),
            _to_float(pref.get("p_scent_level", 3)),
            _to_float(pref.get("p_absorbency_level", 3)),
            _to_float(pref.get("p_adhesion_level", 3)),
            _to_float(pref.get("p_safety_level", 3)),
        ],
        dtype=np.float32,
    )

    vals = (vals - 1.0) / 4.0
    
    seed_src = ",".join(f"{v:.4f}" for v in vals.tolist()).encode("utf-8")
    seed = int(hashlib.sha256(seed_src).hexdigest()[:8], 16)

    rng = np.random.default_rng(seed)

    W = rng.normal(size=(5, dim)).astype(np.float32)
    v = vals @ W

    norm = float(np.linalg.norm(v)) + 1e-12
    v = v / norm
    return v.astype(np.float32).tolist()


@router.get(
    "/personalized-rank",
    summary="키워드별 맞춤 상품 랭킹 조회 API",
)
def personalized_rank(
    memberId: int = Query(..., description="member.id"),
    keyword: Keyword = Query(default="default", description="default | irritationLevel | scent | absorption | adhesion"),
    topK: int = Query(default=TOPK_DEFAULT, ge=1, le=500),
):
    member_id = int(memberId)
    limit_n = int(CANDIDATES_PER_CAT)

    try:
        conn = get_conn()
        try:
            # 1) pref 로드
            with conn.cursor() as cur:
                cur.execute(PREF_SQL, (member_id,))
                pref = cur.fetchone()
                cols = [d[0] for d in cur.description]

            if not pref:
                return {"keyword": keyword, "rankedIds": []}

            pref_dict = pref if isinstance(pref, dict) else dict(zip(cols, pref))

            q_emb = pref_to_query_vec(pref_dict)
            q_emb_str = "[" + ",".join(str(x) for x in q_emb) + "]"

            with conn.cursor() as cur:
                cur.execute(EMB_CANDIDATE_IDS_SQL, (q_emb_str, limit_n))
                id_rows = cur.fetchall()

            product_ids: List[int] = []
            for r in id_rows:
                product_ids.append(r["product_id"] if isinstance(r, dict) else r[0])

            if not product_ids:
                return {"keyword": keyword, "rankedIds": []}

            with conn.cursor() as cur:
                cur.execute(CANDIDATE_BY_IDS_SQL, (member_id, list(product_ids)))
                columns = [d[0] for d in cur.description]
                rows = cur.fetchall()

            df = pd.DataFrame(rows, columns=columns)

        finally:
            conn.close()

    except Exception:
        import traceback
        raise HTTPException(status_code=500, detail=f"DB error: {traceback.format_exc()}")

    if df.empty:
        return {"keyword": keyword, "rankedIds": []}

    try:
        FEATS = model.feature_names
        for c in FEATS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[FEATS] = df[FEATS].fillna(0)
        df["model_score"] = model.score(df)
    except Exception:
        import traceback
        raise HTTPException(status_code=500, detail=traceback.format_exc())

    df = apply_filter_rerank(df, keyword)

    ranked_ids = (
        df.sort_values("final_score", ascending=False)
          .head(topK)["product_id"]
          .astype(int)
          .tolist()
    )

    return {"keyword": keyword, "rankedIds": ranked_ids}