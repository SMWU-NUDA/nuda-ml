from typing import List, Literal
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
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

Keyword = Literal["irritationLevel", "scent", "adhesion", "absorption", "default"]

class GlobalRankRes(BaseModel):
    keyword: Keyword
    rankedIds: List[int] 


def build_score_sql(basis: Keyword) -> str:
    if basis == "default":
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


@router.get(
    "/global-score",
    response_model=GlobalRankRes,
    summary="모든 사용자 대상 키워드 기반 상품추천 API",
)
def global_score(
    keyword: Keyword = Query(default="default", description="default | irritationLevel | scent | adhesion | absorption"),
    topK: int = Query(default=30, ge=1, le=500),
):
    score_sql = build_score_sql(keyword)

    with get_conn() as conn:
        with conn.cursor() as cur:
            sql = f"""
            SELECT
              p.id AS product_id
            FROM v_product_feature_sum v
            JOIN product p
              ON p.external_product_id = v.external_product_id
            ORDER BY ({score_sql})::double precision DESC
            LIMIT %s
            """
            cur.execute(sql, (topK,))
            rows = cur.fetchall()

    ranked_ids = [int(r["product_id"]) for r in rows]
    return GlobalRankRes(keyword=keyword, rankedIds=ranked_ids)