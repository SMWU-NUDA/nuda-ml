# app/routers/keyword_recommend_router.py

from typing import Optional, Literal, List
import os
import pandas as pd

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..db import get_conn
from ..reco_model import RecoModel
from ..sql import CANDIDATE_SQL

router = APIRouter(prefix="/ml/products", tags=["products"])

MODEL_PATH = os.getenv("RECO_MODEL_PATH", "./models/rec_lgbm_rank.txt")
TOPK = int(os.getenv("RECO_TOPK", "20"))
CANDIDATES_PER_CAT = int(os.getenv("RECO_CANDIDATES_PER_CAT", "200"))

model = RecoModel(MODEL_PATH)

Keyword = Literal["all", "irritationLevel", "scent", "absorption", "adhesion"]

class KeywordRecReq(BaseModel):
    memberId: int = Field(..., description="member.id")
    topk: int = Field(default=TOPK, ge=1, le=100)
    keyword: Keyword = Field(default="all")

class ScoreItem(BaseModel):
    productId: int
    score: float

class KeywordRecRes(BaseModel):
    scores: List[ScoreItem]

def d_to_score(d: pd.Series) -> pd.Series:
    return (5 - d).clip(lower=0)

def apply_filter_rerank(df: pd.DataFrame, keyword: Keyword) -> pd.DataFrame:
    if keyword == "all":
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

@router.post(
    "/keyword-recommend",
    response_model=KeywordRecRes,
    summary="사용자별 키워드 기반 점수계산 API",   
    description=(
        "all / irritationLevel / scent / absorption / adhesion 키워드 필터링 가능"
    )
)
def keyword_recommend(req: KeywordRecReq):
    member_id = req.memberId

    # 1) 후보/피처 로드 (CANDIDATE_SQL이 rec_member_pref를 읽어서 후보 재계산하는 구조여야 함)
    try:
        limit_n = int(CANDIDATES_PER_CAT)
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(CANDIDATE_SQL, (member_id, limit_n))
                columns = [d[0] for d in cur.description]
                rows = cur.fetchall()
            df = pd.DataFrame(rows, columns=columns)
        finally:
            conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    if df.empty:
        return {"scores": []}

    # 2) 모델 점수
    try:
        FEATS = model.feature_names
        for c in FEATS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[FEATS] = df[FEATS].fillna(0)
        df["model_score"] = model.score(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # 3) keyword 반영 재정렬
    df = apply_filter_rerank(df, req.keyword)

    out = (
        df.sort_values("final_score", ascending=False)
          .head(req.topk)[["product_id", "final_score"]]
          .rename(columns={"product_id": "productId", "final_score": "score"})
          .to_dict(orient="records")
    )

    return {"scores": out}
