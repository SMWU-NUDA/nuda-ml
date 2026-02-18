import argparse
import pandas as pd
import re
from typing import Tuple, List, Dict, Optional


# "화학물질 특정 불가" 범주형 단어들
UNKNOWN_PATTERNS = [
    (r"(향료|향|프래그런스|퍼퓸|fragrance|perfume)", "fragrance"),
    (r"(접착제|점착제|adhesive|glue|hot\s*melt|핫멜트)", "adhesive"),
    (r"(고분자흡수체|sap|super\s*absorbent|흡수\s*폴리머|흡수체)", "sap"),
    (r"(혼합물|mix|mixture|기타|etc\.?)", "too_generic"),
]

# 너무 광범위해서 MSDS 검색이 의미 없는 소재 단어들
GENERIC_MATERIAL_WORDS = set([
    "부직포", "필름", "시트", "펄프", "면", "순면", "솜", "코튼", "cotton",
    "천연펄프", "레이온", "rayon", "모달", "modal", "텐셀", "tencel",
    "식물성섬유", "면섬유", "레이온스테이플면",
    "패드", "커버", "탑시트", "흡수층", "방수층",
    "날개", "밴드", "코어"
])

# 동의어 매핑
SYNONYMS = {
    "이산화티타늄": "titanium dioxide",
    "티타늄디옥사이드": "titanium dioxide",
    "티타늄 디옥사이드": "titanium dioxide",
    "산화아연": "zinc oxide",
    "폴리에틸렌": "polyethylene",
    "폴리프로필렌": "polypropylene",
    "폴리에스터": "polyester",
}

RE_PAREN = re.compile(r"\([^)]*\)")
RE_PERCENT = re.compile(r"\b\d+(\.\d+)?\s*%")
RE_EXTRA = re.compile(r"[^0-9A-Za-z가-힣\s\-\./,]")

RE_SPLIT = re.compile(r"[,/·;|]+")



def normalize_ingredient_name(raw: str) -> Tuple[str, bool, Optional[str]]:
    if raw is None:
        return "", True, "empty"

    s = str(raw).strip()
    if not s:
        return "", True, "empty"

    s = RE_PAREN.sub(" ", s)
    s = RE_PERCENT.sub(" ", s)
    s = RE_EXTRA.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()

    for pat, reason in UNKNOWN_PATTERNS:
        if re.search(pat, s, flags=re.IGNORECASE):
            return s.lower(), True, reason

    if s in GENERIC_MATERIAL_WORDS:
        return s.lower(), True, "material_generic"

    s2 = s
    for k, v in SYNONYMS.items():
        s2 = s2.replace(k, v)

    s2 = s2.strip().lower()
    if not s2:
        return "", True, "empty"

    return s2, False, None


def extract_tokens(material_name: str, sub_material: str) -> List[str]:
    merged = []
    for x in [material_name, sub_material]:
        if x is None:
            continue
        s = str(x).strip()
        if not s or s.lower() == "nan":
            continue
        merged.append(s)

    if not merged:
        return []

    text = " / ".join(merged)
    text = RE_PAREN.sub(" ", text)
    text = RE_EXTRA.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    parts = RE_SPLIT.split(text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def run_batch(in_path: str, out_norm: str, out_enriched: str):
    df = pd.read_csv(in_path)

    required_cols = ["material_name", "sub_material"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"[ERROR] '{c}' 컬럼이 CSV에 없습니다. 현재 컬럼: {list(df.columns)}")

    all_norm_records = []
    enriched_raw_tokens = []
    enriched_norm_tokens = []
    enriched_unknown_flags = []
    enriched_unknown_reasons = []

    for _, row in df.iterrows():
        tokens = extract_tokens(row.get("material_name"), row.get("sub_material"))
        tokens = dedup_keep_order(tokens)

        row_norm_tokens = []
        row_flags = []
        row_reasons = []

        for t in tokens:
            norm, is_unknown, reason = normalize_ingredient_name(t)

            all_norm_records.append({
                "raw_token": t,
                "normalized_name": norm,
                "is_unknown": is_unknown,
                "unknown_reason": reason
            })

            if norm:  
                row_norm_tokens.append(norm)
                row_flags.append(is_unknown)
                row_reasons.append(reason if reason else "")

        enriched_raw_tokens.append(tokens)
        enriched_norm_tokens.append(dedup_keep_order(row_norm_tokens))
        enriched_unknown_flags.append(row_flags)
        enriched_unknown_reasons.append(row_reasons)

    norm_df = pd.DataFrame(all_norm_records).drop_duplicates(subset=["raw_token"]).reset_index(drop=True)
    norm_df.to_csv(out_norm, index=False, encoding="utf-8-sig")

    df["ingredient_tokens_raw"] = enriched_raw_tokens
    df["ingredient_tokens_norm"] = enriched_norm_tokens
    df["ingredient_unknown_flags"] = enriched_unknown_flags
    df["ingredient_unknown_reasons"] = enriched_unknown_reasons
    df.to_csv(out_enriched, index=False, encoding="utf-8-sig")

    print("[DONE] ingredient_normalized_out.csv saved ->", out_norm)
    print("[DONE] product_ingredient_with_norm.csv saved ->", out_enriched)
    print("[INFO] unique raw_token count =", len(norm_df))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="input csv path")
    parser.add_argument("--out_norm", default="ingredient_normalized_out.csv", help="output normalized summary csv")
    parser.add_argument("--out_enriched", default="product_ingredient_with_norm.csv", help="output enriched csv")
    args = parser.parse_args()

    run_batch(args.in_path, args.out_norm, args.out_enriched)


if __name__ == "__main__":
    main()
