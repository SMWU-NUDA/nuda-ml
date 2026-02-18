# app/routers/user_recommend_router.py

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

FilterKey = Literal["all", "irritationLevel", "scent", "absorption", "adhesion"]

class PersonalRecReq(BaseModel):
    memberId: int = Field(..., description="member.id")
    topk: int = Field(default=TOPK, ge=1, le=100)
    filter: FilterKey = Field(default="all")  # all이면 기존 개인화 추천

class RecItem(BaseModel):
    productId: int
    externalProductId: Optional[str] = None
    score: float

class PersonalRecRes(BaseModel):
    memberId: int
    filter: FilterKey
    items: List[RecItem]

def d_to_score(d: pd.Series) -> pd.Series:
    # d가 작을수록 좋음 -> 큰 점수로 뒤집기
    return (5 - d).clip(lower=0)

def apply_filter_rerank(df: pd.DataFrame, filter_key: FilterKey) -> pd.DataFrame:
    if filter_key == "all":
        df["final_score"] = df["model_score"]
        return df

    col_map = {
        "irritationLevel": "d_sensitivity",
        "scent": "d_scent",
        "absorption": "d_absorbency",
        "adhesion": "d_adhesion",
    }
    dcol = col_map.get(filter_key)
    if not dcol or dcol not in df.columns:
        raise HTTPException(status_code=500, detail=f"missing column: {dcol}")

    df["filter_score"] = d_to_score(pd.to_numeric(df[dcol], errors="coerce").fillna(0))

    # 필터 선택 시: 모델 점수 + 필터점수 가중합
    df["final_score"] = df["model_score"] + df["filter_score"] * 0.2
    return df

@router.post(
    "/personalized-score",
    response_model=PersonalRecRes,
    summary="사용자별 키워드 기반 상품 추천 API",
    description=(
        "사용자별 선호 키워드에 따라 상품을 추천합니다.\n\n"
        "filter = all / irritationLevel / scent / absorption / adhesion 선택 시 해당 키워드 기반으로 모델 점수에 가중치를 주어 재정렬\n\n"
))
def recommend_personal(req: PersonalRecReq):
    member_id = req.memberId

    # 1) 후보/피처 로드
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
        return {"memberId": member_id, "filter": req.filter, "items": []}

    # 2) 모델 점수
    try:
        FEATS = model.feature_names
        for c in FEATS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[FEATS] = df[FEATS].fillna(0)

        df["model_score"] = model.score(df)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {e}")

    # 3) 필터 재정렬
    df = apply_filter_rerank(df, req.filter)

    out = (
        df.sort_values("final_score", ascending=False)
          .head(req.topk)[["product_id", "external_product_id", "final_score"]]
          .rename(columns={
              "product_id": "productId",
              "external_product_id": "externalProductId",
              "final_score": "score",
          })
          .to_dict(orient="records")
    )

    return {"memberId": member_id, "filter": req.filter, "items": out}
