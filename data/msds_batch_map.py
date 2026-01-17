import os
import time
import argparse
import random
from typing import List, Dict, Optional, Tuple
import xml.etree.ElementTree as ET

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import psycopg2
import psycopg2.extras

from dotenv import load_dotenv

load_dotenv() 

KOSHA_CHEMLIST_URL = "https://msds.kosha.or.kr/openapi/service/msdschem/chemlist"


def must_env(name: str) -> str:
    v = os.getenv(name)
    if not v or v.strip() == "":
        raise RuntimeError(f"Missing env var: {name}")
    return v.strip()


def build_dsn() -> str:
    host = must_env("DB_HOST")
    port = must_env("DB_PORT")
    name = must_env("DB_NAME")
    user = must_env("DB_USER")
    pw = must_env("DB_PASSWORD")
    return f"host={host} port={port} dbname={name} user={user} password={pw}"


def make_session() -> requests.Session:
    """
    느린 응답/일시적 오류(429/5xx)에 대비한 retry + backoff 세션.
    """
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.8,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def parse_chemlist(xml_text: str) -> List[Dict]:
    root = ET.fromstring(xml_text)
    items = []
    for it in root.findall(".//item"):
        def t(tag: str) -> str:
            el = it.find(tag)
            return (el.text or "").strip() if el is not None else ""

        items.append(
            {
                "chem_id": t("chemId"),
                "cas_no": t("casNo"),
                "name_kor": t("chemNameKor"),
                "last_date": t("lastDate"),
            }
        )
    return items


def confidence(norm_name: str, cand_name: str, cas_no: str) -> float:
    n = (norm_name or "").strip().lower()
    c = (cand_name or "").strip().lower()
    if not c:
        return 0.0

    score = 0.0
    if n == c:
        score += 0.75
    if n and (n in c or c in n):
        score += 0.15
    if cas_no:
        score += 0.10
    return min(score, 1.0)


def fetch_candidates(
    session: requests.Session,
    api_key: str,
    query: str,
    page_no: int = 1,
    num_rows: int = 30,
) -> Tuple[str, List[Dict]]:
    params = {
        "serviceKey": api_key,
        "searchWrd": query,
        "searchCnd": "0",  # 국문명 검색
        "pageNo": str(page_no),
        "numOfRows": str(num_rows),
    }

   
    r = session.get(KOSHA_CHEMLIST_URL, params=params, timeout=(5, 60))
    r.raise_for_status()
    xml_text = r.text
    print("[DEBUG] query=", query, "items=", len(parse_chemlist(xml_text)), "head=", xml_text[:120])
    return xml_text, parse_chemlist(xml_text)


def upsert_msds_candidates(
    cur,
    ingredient_normalized_id: int,
    xml_text: str,
    cands: List[Dict],
    norm_name: str,
):
    for rank, c in enumerate(cands, start=1):
        conf = confidence(norm_name, c["name_kor"], c["cas_no"])
        cur.execute(
            """
            INSERT INTO msds_candidate (
              ingredient_normalized_id, source, candidate_rank,
              chemical_name, cas_no, msds_id, confidence, raw_payload
            )
            VALUES (%s, 'KOSHA_MSDS', %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ingredient_normalized_id, candidate_rank) DO UPDATE
            SET chemical_name = EXCLUDED.chemical_name,
                cas_no = EXCLUDED.cas_no,
                msds_id = EXCLUDED.msds_id,
                confidence = EXCLUDED.confidence,
                raw_payload = EXCLUDED.raw_payload,
                fetched_at = now()
            """,
            (
                ingredient_normalized_id,
                rank,
                c["name_kor"] or None,
                c["cas_no"] or None,
                c["chem_id"] or None, 
                conf,
                xml_text,
            ),
        )
        print("[DEBUG-UPSERT]", ingredient_normalized_id, "rank", rank, "rowcount", cur.rowcount)



def pick_best(cands: List[Dict], norm_name: str) -> Optional[Dict]:
    if not cands:
        return None

    scored = [(confidence(norm_name, c["name_kor"], c["cas_no"]), c) for c in cands]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    # 보수적 자동선택
    if len(scored) == 1 and best_score >= 0.60:
        best["__score"] = best_score
        return best

    if best_score >= 0.90 and (best_score - second_score) >= 0.20:
        best["__score"] = best_score
        return best

    return None


def create_mapping(
    cur,
    ingredient_normalized_id: int,
    chosen_candidate_id: Optional[int],
    match_status: str,
    match_conf: float,
    decided_by: str,
):
    """
    matched / timeout / error / no_candidate 등 상태 저장 가능하게 확장
    """
    cur.execute(
        """
        INSERT INTO ingredient_msds_mapping (
          ingredient_normalized_id, chosen_candidate_id,
          match_status, match_confidence, decided_by, decided_at
        )
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (ingredient_normalized_id) DO UPDATE
        SET chosen_candidate_id = EXCLUDED.chosen_candidate_id,
            match_status = EXCLUDED.match_status,
            match_confidence = EXCLUDED.match_confidence,
            decided_by = EXCLUDED.decided_by,
            decided_at = EXCLUDED.decided_at
        """,
        (ingredient_normalized_id, chosen_candidate_id, match_status, match_conf, decided_by),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--force", action="store_true", help="이미 매핑된 성분도 재처리(주의)")
    args = ap.parse_args()

    api_key = must_env("KOSHA_MSDS_API_KEY")
    dsn = build_dsn()
    session = make_session()

    conn = psycopg2.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT inet_server_addr(), inet_server_port(), current_database(), version()")
        print("[DB-CONNECT]", cur.fetchone())
        cur.execute("SELECT current_setting('data_directory')")
        print("[DB-DATADIR]", cur.fetchone())
    conn.commit()

    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if args.force:
                cur.execute(
                    """
                    SELECT id, normalized_name
                    FROM ingredient_normalized
                    WHERE is_unknown = false
                    ORDER BY id
                    LIMIT %s
                    """,
                    (args.limit,),
                )
            else:
                cur.execute(
                    """
                    SELECT i.id, i.normalized_name
                    FROM ingredient_normalized i
                    LEFT JOIN ingredient_msds_mapping m
                      ON m.ingredient_normalized_id = i.id
                    WHERE i.is_unknown = false
                      AND m.ingredient_normalized_id IS NULL
                    ORDER BY i.id
                    LIMIT %s
                    """,
                    (args.limit,),
                )

            rows = cur.fetchall()

            for r in rows:
                ing_id = int(r["id"])
                norm_name = str(r["normalized_name"])

                try:
                    xml_text, cands = fetch_candidates(session, api_key, norm_name)
                except requests.exceptions.RequestException as e:
                    # timeout/연결오류 등: 죽지 말고 상태만 기록 후 다음으로
                    print(f"[WARN] fetch failed: ingredient_id={ing_id} name={norm_name} err={e}")
                    create_mapping(
                        cur,
                        ingredient_normalized_id=ing_id,
                        chosen_candidate_id=None,
                        match_status="timeout",
                        match_conf=0.0,
                        decided_by="auto",
                    )
                    conn.commit()
                    time.sleep(args.sleep + random.uniform(0, 0.2))
                    continue

                # 후보 저장(캐시)
                upsert_msds_candidates(cur, ing_id, xml_text, cands, norm_name)

                # 자동선택
                best = pick_best(cands, norm_name)
                if not cands:
                    create_mapping(
                        cur,
                        ingredient_normalized_id=ing_id,
                        chosen_candidate_id=None,
                        match_status="no_candidate",
                        match_conf=0.0,
                        decided_by="auto",
                    )
                elif best:
                    cur.execute(
                        """
                        SELECT id, confidence
                        FROM msds_candidate
                        WHERE ingredient_normalized_id = %s
                          AND msds_id IS NOT DISTINCT FROM %s
                          AND cas_no IS NOT DISTINCT FROM %s
                          AND chemical_name IS NOT DISTINCT FROM %s
                        ORDER BY candidate_rank
                        LIMIT 1
                        """,
                        (
                            ing_id,
                            best.get("chem_id") or None,
                            best.get("cas_no") or None,
                            best.get("name_kor") or None,
                        ),
                    )
                    row2 = cur.fetchone()
                    if row2:
                        chosen_id = int(row2["id"])
                        conf = float(best.get("__score", row2["confidence"] or 0.0))
                        create_mapping(
                            cur,
                            ingredient_normalized_id=ing_id,
                            chosen_candidate_id=chosen_id,
                            match_status="matched",
                            match_conf=conf,
                            decided_by="auto",
                        )
                    else:
                        create_mapping(
                            cur,
                            ingredient_normalized_id=ing_id,
                            chosen_candidate_id=None,
                            match_status="candidate_saved_unmapped",
                            match_conf=0.0,
                            decided_by="auto",
                        )
                else:
                    create_mapping(
                        cur,
                        ingredient_normalized_id=ing_id,
                        chosen_candidate_id=None,
                        match_status="needs_review",
                        match_conf=0.0,
                        decided_by="auto",
                    )

                conn.commit()
                time.sleep(args.sleep + random.uniform(0, 0.15))

        print(f"[DONE] processed {len(rows)} ingredients")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
