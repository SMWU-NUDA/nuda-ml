from datetime import datetime
from typing import Literal

import os
import psycopg
from psycopg.rows import dict_row
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["keyword"])

def get_conn():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        row_factory=dict_row,
        connect_timeout=10,
    )

IrritationLevel = Literal["NONE", "SOMETIMES", "OFTEN"]
ScentLevel = Literal["NONE", "MILD", "STRONG"]
Level3 = Literal["LOW", "MEDIUM", "HIGH"]

class KeywordUpsertRequest(BaseModel):
    memberId: int = Field(..., ge=1)
    irritationLevel: IrritationLevel
    scent: ScentLevel
    changeFrequency: Level3
    thickness: Level3
    adhesion: Level3

class PrefSaved(BaseModel):
    pSensitivityLevel: int
    pScentLevel: int
    pAbsorbencyLevel: int
    pAdhesionLevel: int

class KeywordUpsertResponse(BaseModel):
    memberId: int
    saved: PrefSaved
    asOf: datetime

KEYWORD_UPSERT_SQL = """
INSERT INTO keyword (
    member_id,
    irritation_level,
    scent,
    change_frequency,
    thickness,
    adhesion,
    created_at,
    updated_at
)
VALUES (
    %(member_id)s,
    %(irritation_level)s,
    %(scent)s,
    %(change_frequency)s,
    %(thickness)s,
    %(adhesion)s,
    NOW(),
    NOW()
)
ON CONFLICT (member_id)
DO UPDATE SET
    irritation_level = EXCLUDED.irritation_level,
    scent = EXCLUDED.scent,
    change_frequency = EXCLUDED.change_frequency,
    thickness = EXCLUDED.thickness,
    adhesion = EXCLUDED.adhesion,
    updated_at = NOW();
"""

PREF_UPSERT_SQL = """
INSERT INTO rec_member_pref (
    member_id,
    p_sensitivity_level,
    p_scent_level,
    p_absorbency_level,
    p_adhesion_level,
    p_safety_level,
    as_of
)
VALUES (
    %(member_id)s,

    CASE %(irritation_level)s
        WHEN 'NONE' THEN 1
        WHEN 'SOMETIMES' THEN 3
        WHEN 'OFTEN' THEN 5
        ELSE 3
    END,

    CASE %(scent)s
        WHEN 'NONE' THEN 5
        WHEN 'MILD' THEN 3
        WHEN 'STRONG' THEN 1
        ELSE 3
    END,

    CASE %(change_frequency)s
        WHEN 'LOW' THEN 1
        WHEN 'MEDIUM' THEN 3
        WHEN 'HIGH' THEN 5
        ELSE 3
    END,

    CASE %(adhesion)s
        WHEN 'LOW' THEN 1
        WHEN 'MEDIUM' THEN 3
        WHEN 'HIGH' THEN 5
        ELSE 3
    END,

    COALESCE(
        (SELECT p_safety_level
         FROM rec_member_pref
         WHERE member_id = %(member_id)s),
        3
    ),

    NOW()
)
ON CONFLICT (member_id)
DO UPDATE SET
    p_sensitivity_level = EXCLUDED.p_sensitivity_level,
    p_scent_level = EXCLUDED.p_scent_level,
    p_absorbency_level = EXCLUDED.p_absorbency_level,
    p_adhesion_level = EXCLUDED.p_adhesion_level,
    as_of = NOW()
RETURNING
    member_id,
    p_sensitivity_level,
    p_scent_level,
    p_absorbency_level,
    p_adhesion_level,
    p_safety_level,
    as_of;
"""

@router.post(
    "/members/{memberId}/keyword",
    response_model=KeywordUpsertResponse,
    summary="키워드 저장 및 업데이트 API",
    description="keyword 저장 시 score로 변환하여 저장"
)
def upsert_keyword(memberId: int, req: KeywordUpsertRequest):
    if memberId != req.memberId:
        raise HTTPException(
            status_code=400,
            detail="Path memberId and body memberId do not match"
        )

    params = {
        "member_id": req.memberId,
        "irritation_level": req.irritationLevel,
        "scent": req.scent,
        "change_frequency": req.changeFrequency,
        "thickness": req.thickness,
        "adhesion": req.adhesion,
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(KEYWORD_UPSERT_SQL, params)

                cur.execute(PREF_UPSERT_SQL, params)
                row = cur.fetchone()

                if not row:
                    raise HTTPException(
                        status_code=500,
                        detail="Preference upsert succeeded but no row returned"
                    )

            conn.commit()

        return {
            "memberId": row["member_id"],
            "saved": {
                "pSensitivityLevel": row["p_sensitivity_level"],
                "pScentLevel": row["p_scent_level"],
                "pAbsorbencyLevel": row["p_absorbency_level"],
                "pAdhesionLevel": row["p_adhesion_level"],
            },
            "asOf": row["as_of"],
        }

    except psycopg.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"DB error: {e.pgerror or str(e)}"
        )