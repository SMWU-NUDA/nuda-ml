import os
import time
import argparse
import re
import json
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import psycopg2
import psycopg2.extras
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

load_dotenv()

KOSHA_CHEMDETAIL02_URL = "https://msds.kosha.or.kr/openapi/service/msdschem/chemdetail02"

H_CODE_RE = re.compile(r"\bH\d{3}\b")

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

def build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "nuda-ml/1.0 (+msds hazard fetch)",
        "Accept": "*/*",
    })
    return s

def fetch_chemdetail02(sess: requests.Session, api_key: str, chem_id: str, timeout: int = 30) -> str:
    params = {
        "serviceKey": api_key,
        "chemId": chem_id,
    }
    r = sess.get(KOSHA_CHEMDETAIL02_URL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.text

def parse_chemdetail02(xml_text: str) -> List[Dict]:
    """
    chemdetail02는 'item' 리스트 형태로 내려옴.
    각 item에는 lev, msdsItemCode, upMsdsItemCode, msdsItemNameKor, msdsItemNo, ordrIdx, itemDetail 등이 존재.
    """
    root = ET.fromstring(xml_text)
    items = []
    for it in root.findall(".//item"):
        def t(tag: str) -> str:
            el = it.find(tag)
            return (el.text or "").strip() if el is not None else ""

        item = {
            "lev": t("lev"),
            "msdsItemCode": t("msdsItemCode"),
            "upMsdsItemCode": t("upMsdsItemCode"),
            "msdsItemNameKor": t("msdsItemNameKor"),
            "msdsItemNo": t("msdsItemNo"),
            "ordrIdx": t("ordrIdx"),
            "itemDetail": t("itemDetail"),
        }
        items.append(item)
    return items

def extract_signal_word(items: List[Dict]) -> Optional[str]:
    for it in items:
        name = (it.get("msdsItemNameKor") or "")
        detail = (it.get("itemDetail") or "")
        if "신호어" in name or "신호어" in detail:

            cand = detail.strip()
            if cand:
                return cand
    blob = " ".join((it.get("itemDetail") or "") for it in items)
    for w in ["위험", "경고"]:
        if w in blob:
            return w
    return None

def extract_h_codes(items: List[Dict]) -> List[str]:
    blob = " ".join((it.get("itemDetail") or "") for it in items)
    codes = sorted(set(H_CODE_RE.findall(blob)))
    return codes

def risk_score_from_hcodes(hcodes: List[str], signal_word: Optional[str]) -> int:
    """
    매우 러프한 스코어링(초기 버전).
    나중에 너가 원하는 규칙으로 고도화하면 됨.
    """
    if not hcodes and not signal_word:
        return 0

    score = 0
    if signal_word == "위험":
        score += 20
    elif signal_word == "경고":
        score += 10


    high = {"H300", "H310", "H330", "H340", "H350", "H360", "H370", "H372"}
    mid  = {"H301", "H311", "H331", "H341", "H351", "H361", "H373", "H314", "H318"}
    low  = {"H302", "H312", "H332", "H315", "H319", "H317", "H335", "H336"}

    for c in hcodes:
        if c in high:
            score += 18
        elif c in mid:
            score += 10
        elif c in low:
            score += 5
        else:
            score += 3

    return max(0, min(score, 100))

def risk_level(score: int) -> str:
    if score >= 70:
        return "high"
    if score >= 35:
        return "medium"
    if score > 0:
        return "low"
    return "unknown"

def upsert_hazard(cur, chosen_candidate_id: int, signal_word: Optional[str], hcodes: List[str], items: List[Dict], score: int, level: str):
    cur.execute(
        """
        INSERT INTO msds_hazard (
          chosen_candidate_id,
          signal_word,
          h_codes,
          hazard_classes,
          risk_score,
          risk_level,
          parsed_at
        )
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, now())
        ON CONFLICT (chosen_candidate_id) DO UPDATE
        SET signal_word = EXCLUDED.signal_word,
            h_codes = EXCLUDED.h_codes,
            hazard_classes = EXCLUDED.hazard_classes,
            risk_score = EXCLUDED.risk_score,
            risk_level = EXCLUDED.risk_level,
            parsed_at = now()
        """,
        (
            chosen_candidate_id,
            signal_word,
            json.dumps(hcodes, ensure_ascii=False),
            json.dumps(items, ensure_ascii=False),
            score,
            level,
        )
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--force", action="store_true", help="이미 msds_hazard가 있는 후보도 재처리")
    args = ap.parse_args()

    api_key = must_env("KOSHA_MSDS_API_KEY")
    dsn = build_dsn()
    sess = build_session()

    conn = psycopg2.connect(dsn)
    conn.autocommit = False

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if args.force:
                cur.execute(
                    """
                    SELECT
                      m.chosen_candidate_id,
                      c.msds_id AS chem_id
                    FROM ingredient_msds_mapping m
                    JOIN msds_candidate c ON c.id = m.chosen_candidate_id
                    WHERE m.match_status = 'matched'
                      AND c.msds_id IS NOT NULL
                    ORDER BY m.id
                    LIMIT %s
                    """,
                    (args.limit,)
                )
            else:
                cur.execute(
                    """
                    SELECT
                      m.chosen_candidate_id,
                      c.msds_id AS chem_id
                    FROM ingredient_msds_mapping m
                    JOIN msds_candidate c ON c.id = m.chosen_candidate_id
                    LEFT JOIN msds_hazard h ON h.chosen_candidate_id = m.chosen_candidate_id
                    WHERE m.match_status = 'matched'
                      AND c.msds_id IS NOT NULL
                      AND h.chosen_candidate_id IS NULL
                    ORDER BY m.id
                    LIMIT %s
                    """,
                    (args.limit,)
                )

            rows = cur.fetchall()
            processed = 0

            for r in rows:
                chosen_id = int(r["chosen_candidate_id"])
                chem_id = str(r["chem_id"]).strip()

                xml_text = fetch_chemdetail02(sess, api_key, chem_id, timeout=30)
                items = parse_chemdetail02(xml_text)

                sig = extract_signal_word(items)
                hcodes = extract_h_codes(items)

                score = risk_score_from_hcodes(hcodes, sig)
                level = risk_level(score)

                upsert_hazard(cur, chosen_id, sig, hcodes, items, score, level)

                conn.commit()
                processed += 1
                time.sleep(args.sleep)

        print(f"[DONE] hazard processed {processed} candidates")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
