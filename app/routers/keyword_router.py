from typing import List, Optional, Literal
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from psycopg import connect
from psycopg.rows import dict_row

router = APIRouter(prefix="/ml/products", tags=["products"])

def get_conn():
    return connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        row_factory=dict_row,
    )

Keyword = Literal["irritationLevel", "scent", "adhesion", "absorption", "all"]

class GlobalScoreReq(BaseModel):
    keyword: Optional[Keyword] = Field(default="all")
    limit: int = Field(default=50, ge=1, le=200)


class ProductRank(BaseModel):
    productId: int
    score: float


class GlobalScoreRes(BaseModel):
    rankingBasis: Optional[str]
    products: List[ProductRank]
    analyzedAt: datetime


def build_score_sql(basis: Keyword) -> str:
    if basis == "all":
        return """
        (COALESCE(v.sensitivity_sum,0)
       + COALESCE(v.scent_sum,0)
       + COALESCE(v.absorbency_sum,0)
       + COALESCE(v.adhesion_sum,0))
        """
    if basis == "irritationLevel":
        return "COALESCE(v.sensitivity_sum,0)"
    if basis == "scent":
        return "COALESCE(v.scent_sum,0)"
    if basis == "adhesion":
        return "COALESCE(v.adhesion_sum,0)"
    if basis == "absorption":
        return "COALESCE(v.absorbency_sum,0)"
    raise HTTPException(status_code=400, detail=f"unsupported keyword: {basis}")


@router.post(
    "/global-score",
    response_model=GlobalScoreRes,
    summary="모든 사용자 대상 키워드 기반 상품추천 API",
)
def global_score(req: GlobalScoreReq):

    basis = req.keyword or "all"
    score_sql = build_score_sql(basis)
    ranking_basis_out = None if basis == "all" else basis

    with get_conn() as conn:
        with conn.cursor() as cur:

            sql = f"""
            SELECT
              p.id AS product_id,
              ({score_sql})::double precision AS score
            FROM v_product_feature_sum v
            JOIN product p
              ON p.external_product_id = v.external_product_id
            ORDER BY score DESC
            LIMIT %s
            """

            cur.execute(sql, (req.limit,))
            rows = cur.fetchall()

    return GlobalScoreRes(
        rankingBasis=ranking_basis_out,
        products=[
            ProductRank(
                productId=r["product_id"],
                score=float(r["score"])
            ) for r in rows
        ],
        analyzedAt=datetime.now(timezone.utc),
    )
