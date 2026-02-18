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
from urllib.parse import unquote

load_dotenv()

KOSHA_CHEMDETAIL02_URL = "https://msds.kosha.or.kr/openapi/service/msdschem/chemdetail02"

H_CODE_RE = re.compile(r"\bH\d{3}\b")
P_CODE_RE = re.compile(r"\bP\d{3}(?:\+P\d{3})?\b")

EMPTY_MARKERS = {"자료없음", "없음", "해당없음", "-", "N/A", "na", "n/a"}

# B02 텍스트(유해성·위험성 분류)에서 등장하는 분류명을 내부 key로 매핑
B02_MAP = {
    "발암성": "carcinogenicity",
    "생식독성": "reprotox",
    "변이원성": "mutagenicity",
    "생식세포 변이원성": "mutagenicity",
    "특정표적장기 독성(단회 노출)": "stot_se",
    "특정표적장기독성(단회노출)": "stot_se",
    "특정표적장기독성(단회 노출)": "stot_se",
    "특정표적장기 독성(반복 노출)": "stot_re",
    "특정표적장기독성(반복노출)": "stot_re",
    "특정표적장기독성(반복 노출)": "stot_re",
    "급성독성(경구)": "acute_oral",
    "급성독성(경피)": "acute_dermal",
    "급성독성(흡입)": "acute_inhal",
}
B02_MAP.update({
    # 피부/눈
    "피부 부식성/피부 자극성": "skin_irrit",
    "심한 눈 손상성/눈 자극성": "eye_irrit",

    # 과민성
    "피부 과민성": "skin_sens",
    "호흡기 과민성": "resp_sens",

    # 흡인/수생
    "흡인 유해성": "asp_hazard",
    "급성 수생환경 유해성": "aquatic_acute",
    "만성 수생환경 유해성": "aquatic_chronic",
})

# H-code 기반 severity (fallback/보조)
H_SEVERITY: Dict[str, Tuple[str, int]] = {
    # very high
    "H300": ("acute_fatal", 90),
    "H310": ("acute_fatal", 90),
    "H330": ("acute_fatal", 90),
    "H340": ("mutagenicity", 88),
    "H350": ("carcinogenicity", 88),
    "H360": ("reprotox", 88),
    "H370": ("stot_se_1", 80),
    "H372": ("stot_re_1", 78),

    # high
    "H301": ("acute_tox", 70),
    "H311": ("acute_tox", 70),
    "H331": ("acute_tox", 70),
    "H341": ("mutagenicity_2", 65),
    "H351": ("carcinogenicity_2", 65),
    "H361": ("reprotox_2", 65),
    "H373": ("stot_re_2", 60),
    "H314": ("skin_corr", 60),
    "H318": ("eye_damage", 55),

    # medium/low
    "H302": ("acute_tox_4", 45),
    "H312": ("acute_tox_4", 45),
    "H332": ("acute_tox_4", 45),
    "H315": ("skin_irrit", 25),
    "H319": ("eye_irrit", 25),
    "H317": ("skin_sens", 30),
    "H335": ("resp_irrit", 28),
    "H336": ("narcotic", 20),
}


def must_env(name: str) -> str:
    v = os.getenv(name)
    if not v or v.strip() == "":
        raise RuntimeError(f"Missing env var: {name}")
    return v.strip()


def normalize_service_key(k: str) -> str:
    # data.go.kr 계열은 인코딩된 키를 주는 경우가 있어 1회 디코딩
    return unquote((k or "").strip())


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
    items: List[Dict] = []
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


def clean_text(x: str) -> str:
    x = (x or "").strip()
    # 줄바꿈/탭 등 공백을 한 칸으로 정리 (UI/검색/파싱 안정화)
    x = re.sub(r"\s+", " ", x)
    return "" if x in EMPTY_MARKERS else x


def index_items_by_code(items: List[Dict]) -> Dict[str, List[Dict]]:
    d: Dict[str, List[Dict]] = {}
    for it in items:
        code = (it.get("msdsItemCode") or "").strip()
        if not code:
            continue
        d.setdefault(code, []).append(it)
    return d


def extract_core_fields(items: List[Dict]) -> Dict:
    """
    점수 산정 및 UI 설명에 필요한 핵심 필드들만 구조화해서 뽑는다.
    - B02: 유해성·위험성 분류 (구분1/2 등)
    - B0402: 그림문자
    - B0404: 신호어
    - B0406: 유해·위험문구(H문구)
    - B0408 하위: 예방/대응/저장/폐기(P문구)
    """
    by_code = index_items_by_code(items)

    def details(code: str) -> List[str]:
        out: List[str] = []
        for it in by_code.get(code, []):
            out.append(clean_text(it.get("itemDetail") or ""))
        return [x for x in out if x]

    # B02
    class_texts = details("B02")

    # pictograms (B0402) - 파일명 형태가 많음
    pictograms = details("B0402")

    # signal word (B0404)
    signal_word = None
    b0404 = details("B0404")
    if b0404:
        sw = b0404[0].strip()
        if sw in ("위험", "경고"):
            signal_word = sw

    # hazard statements (B0406) - H-code & 문구
    hazard_statements = details("B0406")

    # precautionary statements (B0408 하위)
    p_prevention = details("B040802")
    p_response = details("B040804")
    p_storage = details("B040806")
    p_disposal = details("B040808")

    return {
        "hazard_classes_text": class_texts,
        "pictograms": pictograms,
        "signal_word": signal_word,
        "hazard_statements": hazard_statements,
        "precautionary": {
            "prevention": p_prevention,
            "response": p_response,
            "storage": p_storage,
            "disposal": p_disposal,
        },
    }


def extract_h_codes_from_statements(hazard_statements: List[str]) -> List[str]:
    blob = " ".join(hazard_statements)
    return sorted(set(H_CODE_RE.findall(blob)))


def parse_h_statements(hazard_statements: List[str]) -> List[Dict]:
    """
    "H351 : ....|H372 : ...." 형태를 [{"code":"H351","text":"..."}, ...] 로 분해
    """
    out: List[Dict] = []
    for line in hazard_statements:
        parts = [p.strip() for p in line.split("|") if p.strip()]
        for p in parts:
            # H351 : text
            m = re.match(r"(H\d{3})\s*:\s*(.+)$", p)
            if m:
                out.append({"code": m.group(1), "text": m.group(2).strip()})
            else:
                # 형식이 다르면 그냥 원문 보관
                out.append({"code": None, "text": p})
    # code 기준 정렬(있으면)
    out.sort(key=lambda x: (x["code"] is None, x["code"] or "", x["text"]))
    return out


def parse_p_statements(precautionary: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
    """
    "P201 : ....|P202 : ...." 형태를 [{"code":"P201","text":"..."}, ...] 로 분해
    """
    def split_lines(lines: List[str]) -> List[Dict]:
        out: List[Dict] = []
        for line in lines:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            for p in parts:
                m = re.match(r"(P\d{3}(?:\+P\d{3})?)\s*:\s*(.+)$", p)
                if m:
                    out.append({"code": m.group(1), "text": m.group(2).strip()})
                else:
                    out.append({"code": None, "text": p})
        out.sort(key=lambda x: (x["code"] is None, x["code"] or "", x["text"]))
        return out

    return {
        "prevention": split_lines(precautionary.get("prevention", [])),
        "response": split_lines(precautionary.get("response", [])),
        "storage": split_lines(precautionary.get("storage", [])),
        "disposal": split_lines(precautionary.get("disposal", [])),
    }


def parse_b02_detail(detail: str) -> List[Dict]:
    """
    예) "발암성 : 구분2|특정표적장기 독성(반복 노출) : 구분1"
    -> [{"class":"carcinogenicity","category":"2","raw":"..."}, {"class":"stot_re","category":"1","raw":"..."}]
    """
    detail = clean_text(detail)
    if not detail:
        return []
    parts = [p.strip() for p in detail.split("|") if p.strip()]
    out: List[Dict] = []
    for p in parts:
        m = re.match(r"(.+?)\s*:\s*구분\s*(\d+)", p)
        if not m:
            continue
        raw_name = m.group(1).strip()
        cat = m.group(2).strip()

        # raw_name이 약간씩 달라질 수 있으니 매핑을 최대한 유연하게
        cls = B02_MAP.get(raw_name)
        if not cls:
            # 공백/특수문자 차이를 줄여서 재시도
            key_norm = re.sub(r"\s+", " ", raw_name)
            cls = B02_MAP.get(key_norm)

        if cls:
            out.append({"class": cls, "category": cat, "raw": p})
        else:
            out.append({"class": None, "category": cat, "raw": p})
    return out


def extract_categories_from_b02_texts(class_texts: List[str]) -> List[Dict]:
    out: List[Dict] = []
    for t in class_texts:
        out.extend(parse_b02_detail(t))
    # class가 None인 항목은 점수엔 쓰지 않지만 근거/디버깅용으로 남길 수 있음
    return out


def score_from_categories(categories: List[Dict]) -> Tuple[Optional[int], Optional[Dict]]:
    """
    category 기반 worst-case 점수.
    B02의 "구분1/2/3/4"를 category 숫자로 보고,
    class별 위험도를 반영한다.
    """
    usable = [c for c in categories if c.get("class") and c.get("category")]
    if not usable:
        return None, None

    max_points = {
        "carcinogenicity": 88,
        "mutagenicity": 88,
        "reprotox": 88,
        "acute_oral": 90,
        "acute_dermal": 90,
        "acute_inhal": 90,
        "stot_se": 80,
        "stot_re": 78,
    }
    max_points.update({
    "skin_irrit": 40,
    "eye_irrit": 40,
    "skin_sens": 45,
    "resp_sens": 55,
    "asp_hazard": 75,
    "aquatic_acute": 35,
    "aquatic_chronic": 30,
})


    def ratio(cls: str, cat: str) -> float:
        # 구분 숫자가 작을수록 더 위험
        # (GHS 카테고리 구조를 단순화한 버전)
        if cat == "1":
            return 1.0
        if cat == "2":
            return 0.75 if cls in ("carcinogenicity", "mutagenicity", "reprotox", "stot_se", "stot_re") else 0.55
        if cat == "3":
            return 0.35
        if cat == "4":
            return 0.20
        return 0.0

    best_score: Optional[int] = None
    best_entry: Optional[Dict] = None

    for c in usable:
        cls = str(c["class"])
        cat = str(c["category"])
        base = max_points.get(cls, 0)
        s = int(round(base * ratio(cls, cat)))
        if best_score is None or s > best_score:
            best_score = s
            best_entry = {"basis": "category", "class": cls, "category": cat, "base": base, "score": s, "raw": c.get("raw")}

    return best_score, best_entry


def score_and_basis(
    signal_word: Optional[str],
    hcodes: List[str],
    categories: List[Dict],
    pictograms: List[str],
) -> Tuple[int, str, float, Optional[str], Dict]:
    """
    최종 점수/레벨 + confidence + unknown_reason + risk_basis(근거) 반환
    - 우선순위: category(B02) > H-code(B0406)
    - signal_word/pictogram은 보정 및 근거로만 사용
    """
    cat_score, cat_basis = score_from_categories(categories)

    worst_h = None
    worst_h_score = None
    worst_h_basis = None
    for h in hcodes:
        if h in H_SEVERITY:
            typ, s = H_SEVERITY[h]
            if worst_h_score is None or s > worst_h_score:
                worst_h_score = s
                worst_h = h
                worst_h_basis = {"basis": "hcode", "worst_h": h, "type": typ, "score": s}

    # base 선택
    if cat_score is not None:
        base_score = cat_score
        selected_basis = cat_basis or {"basis": "category"}
    elif worst_h_score is not None:
        base_score = worst_h_score
        selected_basis = worst_h_basis or {"basis": "hcode"}
    else:
        # 정보 없음
        confidence = 0.0
        unknown_reason = "no_hazard_data"
        risk_basis = {
            "selected_basis": {"basis": "none"},
            "signals": {"signal_word": signal_word, "pictograms": pictograms[:10]},
            "inputs": {"h_codes": hcodes, "categories": categories},
            "bonuses": {"signal_word_bonus": 0, "extra_hcode_bonus": 0},
            "final_score": 0,
            "confidence": confidence,
            "unknown_reason": unknown_reason,
        }
        return 0, "unknown", confidence, unknown_reason, risk_basis

    # 보정: signal_word
    signal_bonus = 0
    if signal_word == "위험":
        signal_bonus = 5
    elif signal_word == "경고":
        signal_bonus = 2

    # 보정: 추가 H-code (worst 제외) 존재 기반 최대 +10
    extra = 0
    if hcodes:
        others = []
        for h in hcodes:
            if h == worst_h:
                continue
            if h in H_SEVERITY:
                others.append(H_SEVERITY[h][1])
        others.sort(reverse=True)
        # 존재 기반 보정(설명하기 쉬움)
        if others:
            extra += 3
        if len(others) >= 2:
            extra += 3
        if len(others) >= 3:
            extra += 2
        if len(others) >= 4:
            extra += 2
        extra = min(extra, 10)

    final_score = max(0, min(int(base_score + signal_bonus + extra), 100))

    # confidence
    confidence = 0.0
    if cat_score is not None:
        confidence += 0.65
        if worst_h_score is not None:
            confidence += 0.15
    elif worst_h_score is not None:
        confidence += 0.45

    if signal_word in ("위험", "경고"):
        confidence += 0.10
    if pictograms:
        confidence += 0.05
    confidence = min(confidence, 0.95)

    # level
    if confidence < 0.35:
        level = "unknown"
    elif final_score >= 70:
        level = "high"
    elif final_score >= 35:
        level = "medium"
    elif final_score > 0:
        level = "low"
    else:
        level = "unknown"

    unknown_reason = None

    risk_basis = {
        "selected_basis": selected_basis,
        "signals": {
            "signal_word": signal_word,
            "pictograms": pictograms[:10],
        },
        "inputs": {
            "h_codes": hcodes,
            "categories": categories,
        },
        "bonuses": {
            "signal_word_bonus": signal_bonus,
            "extra_hcode_bonus": extra,
        },
        "final_score": final_score,
        "confidence": confidence,
        "unknown_reason": unknown_reason,
    }

    return final_score, level, confidence, unknown_reason, risk_basis


def upsert_hazard(
    cur,
    chosen_candidate_id: int,
    signal_word: Optional[str],
    hcodes: List[str],
    items: List[Dict],
    score: int,
    level: str,
    confidence: float,
    unknown_reason: Optional[str],
    risk_basis: Dict,
    extracted: Dict,
):
    """
    msds_hazard 테이블에 아래 컬럼이 있다고 가정:
      - confidence REAL
      - unknown_reason TEXT
      - risk_basis JSONB
      - extracted JSONB
    (없으면 ALTER TABLE로 추가 필요)
    """
    cur.execute(
        """
        INSERT INTO msds_hazard (
          chosen_candidate_id,
          signal_word,
          h_codes,
          hazard_classes,
          risk_score,
          risk_level,
          confidence,
          unknown_reason,
          risk_basis,
          extracted,
          parsed_at
        )
        VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s::jsonb, %s::jsonb, now())
        ON CONFLICT (chosen_candidate_id) DO UPDATE
        SET signal_word = EXCLUDED.signal_word,
            h_codes = EXCLUDED.h_codes,
            hazard_classes = EXCLUDED.hazard_classes,
            risk_score = EXCLUDED.risk_score,
            risk_level = EXCLUDED.risk_level,
            confidence = EXCLUDED.confidence,
            unknown_reason = EXCLUDED.unknown_reason,
            risk_basis = EXCLUDED.risk_basis,
            extracted = EXCLUDED.extracted,
            parsed_at = now()
        """,
        (
            chosen_candidate_id,
            signal_word,
            json.dumps(hcodes, ensure_ascii=False),
            json.dumps(items, ensure_ascii=False),  # 원본 items 전체 보관(설명/검증용)
            score,
            level,
            confidence,
            unknown_reason,
            json.dumps(risk_basis, ensure_ascii=False),
            json.dumps(extracted, ensure_ascii=False),
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--force", action="store_true", help="이미 msds_hazard가 있는 후보도 재처리")
    args = ap.parse_args()

    api_key = normalize_service_key(must_env("KOSHA_MSDS_API_KEY"))
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

                core = extract_core_fields(items)

                signal_word = core["signal_word"]
                hazard_statements = core["hazard_statements"]
                class_texts = core["hazard_classes_text"]
                pictograms = core["pictograms"]
                precautionary = core["precautionary"]

                hcodes = extract_h_codes_from_statements(hazard_statements)
                categories = extract_categories_from_b02_texts(class_texts)

                score, level, confidence, unknown_reason, risk_basis = score_and_basis(
                    signal_word=signal_word,
                    hcodes=hcodes,
                    categories=categories,
                    pictograms=pictograms,
                )

                extracted = {
                    "signal_word": signal_word,
                    "pictograms": pictograms,
                    "hazard_classes_text": class_texts,
                    "categories": categories,
                    "h_statements": parse_h_statements(hazard_statements),
                    "p_statements": parse_p_statements(precautionary),
                }

                upsert_hazard(
                    cur,
                    chosen_id,
                    signal_word,
                    hcodes,
                    items,
                    score,
                    level,
                    confidence,
                    unknown_reason,
                    risk_basis,
                    extracted,
                )

                conn.commit()
                processed += 1
                time.sleep(args.sleep)

        print(f"[DONE] hazard processed {processed} candidates")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
