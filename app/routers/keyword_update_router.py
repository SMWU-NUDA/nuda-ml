from datetime import datetime
from typing import Literal, Optional, Dict, Any

import os
import psycopg
from psycopg.rows import dict_row
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["update"])

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
ScentLevel      = Literal["NONE", "MILD", "STRONG"]
Level3          = Literal["LOW", "MEDIUM", "HIGH"]

class PrefUpsertRequest(BaseModel):
    memberId: int = Field(..., ge=1)
    irritationLevel: IrritationLevel
    scent: ScentLevel
    absorption: Level3
    adhesion: Level3

class PrefSaved(BaseModel):
    pSensitivityLevel: int
    pScentLevel: int
    pAbsorbencyLevel: int
    pAdhesionLevel: int

class PrefUpsertResponse(BaseModel):
    memberId: int
    saved: PrefSaved
    asOf: datetime

UPSERT_SQL = """
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

    CASE %(irritationLevel)s
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

    CASE %(absorption)s
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
      (SELECT p_safety_level FROM rec_member_pref WHERE member_id = %(member_id)s),
      3
    ),

    NOW()
)
ON CONFLICT (member_id)
DO UPDATE SET
    p_sensitivity_level = EXCLUDED.p_sensitivity_level,
    p_scent_level       = EXCLUDED.p_scent_level,
    p_absorbency_level  = EXCLUDED.p_absorbency_level,
    p_adhesion_level    = EXCLUDED.p_adhesion_level,
    as_of               = NOW()
RETURNING
    member_id,
    p_sensitivity_level,
    p_scent_level,
    p_absorbency_level,
    p_adhesion_level,
    p_safety_level,
    as_of;
"""

@router.post("/members/{memberId}/preference/keyword-update", response_model=PrefUpsertResponse,
             summary="키워드 자동 업데이트 API", description="회원 키워드 업데이트시 점수로 변환해서 자동 저장")
def upsert_member_pref(req: PrefUpsertRequest):
    params = {
        "member_id": req.memberId,
        "irritationLevel": req.irritationLevel,
        "scent": req.scent,
        "absorption": req.absorption,
        "adhesion": req.adhesion,
    }

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(UPSERT_SQL, params)
                row = cur.fetchone()
                if not row:
                    raise HTTPException(status_code=500, detail="Upsert succeeded but no row returned")

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
        raise HTTPException(status_code=500, detail=f"DB error: {e.pgerror or str(e)}")