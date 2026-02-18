# review_keyword_deploy.py
import os, re, json, random, unicodedata, math
from collections import Counter, defaultdict
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

load_dotenv()

# =======================
# DB
# =======================
def db_conn():
    return psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )

# =======================
# Model
# =======================
MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
st_model = SentenceTransformer(MODEL_NAME, device="cpu")
kw_model = KeyBERT(model=st_model)

# =======================
# Settings
# =======================
MIN_REVIEWS_FOR_KEYWORDS = 10
CHUNK_SIZE = 200
MAX_REVIEW_CHARS = 220
MAX_DOC_CHARS = 12000

TOP_N_KEYPHRASES = 50
CANDIDATE_MULTIPLIER = 4

TOP_N_KEYWORDS = 30
TOP_N_ASPECT_KEYWORDS = 12

STOPWORDS = {
    "배송","포장","가격","쿠폰","사은품","할인","구매","주문","재구매","추천","만족",
    "제품","상품","사용","이번","항상","진짜","너무","그냥","완전","약간","정말",
    "올리브영","올영","마트","행사","세일","증정","무료","구입",
    "기획","패키지","구성","브랜드","쓰는","세면","생리대"
}

# -----------------------
# Clean text
# -----------------------
CTRL_RE = re.compile(
    r"[\u0000-\u001F\u007F]"
    r"|[\u200B-\u200F\u202A-\u202E]"
    r"|[\u2028\u2029]"
)
VS_RE = re.compile(r"[\uFE0E\uFE0F]")
EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001F5FF" "\U0001F600-\U0001F64F" "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F" "\U0001F780-\U0001F7FF" "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF" "\U0001FA00-\U0001FAFF" "\u2600-\u26FF" "\u2700-\u27BF"
    "\u2B00-\u2BFF" "]+",
    flags=re.UNICODE
)

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = CTRL_RE.sub(" ", s)
    s = VS_RE.sub("", s)
    s = EMOJI_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------
# Aspect rules (✅ 향/냄새 강화)
# -----------------------
ASPECT_RULES = {
    "흡수/샘": ["흡수","흡수력","샘","새","샜","새요","새는","새지","샘방지","양많","오버나이트","밤","많은날","역류","뭉침"],
    "민감/자극": ["자극","가려","가렵","따가","따갑","트러블","민감","발진","간지","피부","가려움","가렵지","자극없","순하","저자극","편안","부드럽"],
    "향/냄새": [  # ✅ 확장
        "향","냄새","무향","향료","향이","냄새가","비린","비릿","잡냄새","체취","향기","향긋"
    ],
    "두께/슬림": ["얇","얇아","슬림","두께","두껍","도톰","두툼","도톰하","티안","티나","부피","휴대","가볍"],
    "착용감": ["편안","편하","편해","쾌적","가볍","부드럽","촉감","보송","산뜻","불편","이물감","착용","답답","움직","말림","구겨","쓸림","쓸리","밀착"],
    "접착/고정": ["접착","고정","고정력","잘붙","붙어","떨어","안떨어","밀림","밀려","움직임","날개","뜯","찢","접힘"],
    "사이즈/핏": ["대형","중형","소형","롱","팬티라이너","길이","사이즈","핏","너비","폭","커버","면적","크기"],
}

def map_phrase_to_aspect(phrase: str) -> Optional[str]:
    p = phrase.replace(" ", "")
    for aspect, kws in ASPECT_RULES.items():
        for k in kws:
            if k in p:
                return aspect
    return None

# -----------------------
# sentiment ratios (1~5)
# -----------------------
def sentiment_ratio(scores):
    s = pd.to_numeric(pd.Series(scores), errors="coerce").dropna()
    if len(s) == 0:
        return 0.0, 0.0, 0.0
    pos = float((s >= 4).mean())
    neu = float((s == 3).mean())
    neg = float((s <= 2).mean())
    return pos, neu, neg

# -----------------------
# keyphrase filtering
# -----------------------
BAD_PATTERNS = re.compile(r"(\d{1,2}월|\d{1,2}일|\d{4}|\d+%|원|까지|담달|행사|세일|올영|올리브영|마트|쿠폰|가격|포장|배송)")
JOSA_ENDINGS = ("이다","네요","해요","합니다","같아요","있어요","없어요","좋아요","싫어요","했어요","되네요","되요","에요","입니다")

def is_good_keyphrase(ph: str) -> bool:
    ph = ph.strip()
    if len(ph) < 2 or len(ph) > 20:
        return False
    if BAD_PATTERNS.search(ph):
        return False
    if any(ph.endswith(e) for e in JOSA_ENDINGS):
        return False
    digits = sum(c.isdigit() for c in ph)
    if digits >= 3:
        return False
    if ph.count(" ") >= 4:
        return False
    toks = ph.split()
    if len(toks) == 1 and toks[0] in STOPWORDS:
        return False
    return True

def chunk_reviews(reviews: List[str], chunk_size: int) -> List[List[str]]:
    return [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]

def extract_keyphrases_all(reviews: List[str], top_n=TOP_N_KEYPHRASES) -> Tuple[List[str], Dict[str, float]]:
    phrase_scores = Counter()
    if not reviews:
        return [], {}

    reviews_shuffled = reviews[:]
    random.shuffle(reviews_shuffled)

    chunks = chunk_reviews(reviews_shuffled, CHUNK_SIZE)
    for chunk in chunks:
        chunk = [r[:MAX_REVIEW_CHARS] for r in chunk if r]
        doc = " ".join(chunk)[:MAX_DOC_CHARS]
        if not doc.strip():
            continue

        kws = kw_model.extract_keywords(
            doc,
            keyphrase_ngram_range=(1, 3),
            stop_words=list(STOPWORDS),
            top_n=top_n * CANDIDATE_MULTIPLIER,
            use_mmr=True,
            diversity=0.65,
        )

        for phrase, score in kws:
            if not phrase or score is None:
                continue
            phrase = phrase.strip()
            if not is_good_keyphrase(phrase):
                continue
            score = float(score)
            if math.isnan(score) or math.isinf(score):
                continue
            phrase_scores[phrase] += score

    top_phrases = [p for p, _ in phrase_scores.most_common(top_n)]
    return top_phrases, dict(phrase_scores)

# -----------------------
# phrase -> token keywords (weighted)
# -----------------------
TOKEN_BAD = re.compile(r"(\d|%|원|까지|담달|행사|세일|올영|올리브영|배송|포장|쿠폰|가격)")
KOREAN_JOSA_END = ("은","는","이","가","을","를","에","에서","에게","한테","로","으로","와","과","랑","하고","도","만","까지","부터","마다","이나","나","처럼","같이","보다","의","께","뿐","조차","마저")
COLLOQUIAL_STOP = {"좋더라구요","좋아요","좋네","좋음","짱","짱짱","대박","추천","만족","쓰는중","써요","사용해요","사용중","계속","항상","매번","그냥","진짜","너무","완전","약간","정말","괜찮아요","괜찮음","입니당","이에요","예요","네요","해요","했어요","되요","돼요","것","거","부분","이것","저것"}
ENDING_TRIMS = ("입니다","이에요","예요","네요","해요","했어요","했네","했음","합니다","같아요","있어요","없어요","되요","돼요","임","요")
MIN_KW_LEN = 2

def strip_josa(token: str) -> str:
    for j in sorted(KOREAN_JOSA_END, key=len, reverse=True):
        if token.endswith(j) and len(token) > len(j) + 1:
            return token[:-len(j)]
    return token

def normalize_token(token: str) -> str:
    t = token.strip()
    if not t:
        return ""
    for e in ENDING_TRIMS:
        if t.endswith(e) and len(t) > len(e) + 1:
            t = t[:-len(e)]
            break
    t = strip_josa(t)
    t = re.sub(r"^[^\w가-힣]+|[^\w가-힣]+$", "", t)
    return t

def phrase_to_keywords(phrase: str) -> List[str]:
    phrase = clean_text(phrase)
    toks = [t for t in phrase.split() if t]
    out = []
    for raw in toks:
        if TOKEN_BAD.search(raw):
            continue
        if raw in STOPWORDS or raw in COLLOQUIAL_STOP:
            continue
        t = normalize_token(raw)
        if not t:
            continue
        if t in STOPWORDS or t in COLLOQUIAL_STOP:
            continue
        if len(t) < MIN_KW_LEN or len(t) > 10:
            continue
        if t.endswith(("하", "되", "있", "없")):
            continue
        out.append(t)

    seen = set()
    dedup = []
    for t in out:
        if t not in seen:
            seen.add(t)
            dedup.append(t)
    return dedup

def build_keywords_from_keyphrases_weighted(
    keyphrases: List[str],
    phrase_score_map: Dict[str, float],
    alpha_len_norm: float = 0.7,
) -> Tuple[List[str], Dict[str, List[str]]]:
    keyword_scores = Counter()
    aspect_keywords = defaultdict(Counter)

    for ph in keyphrases:
        ph_score = float(phrase_score_map.get(ph, 0.0))
        if ph_score <= 0:
            continue

        aspect = map_phrase_to_aspect(ph)
        toks = phrase_to_keywords(ph)
        if not toks:
            continue

        len_norm = (1.0 / (len(toks) ** alpha_len_norm))
        per_tok = ph_score * len_norm

        for t in toks:
            keyword_scores[t] += per_tok
            if aspect:
                aspect_keywords[aspect][t] += per_tok

    top_keywords = [k for k, _ in keyword_scores.most_common(TOP_N_KEYWORDS)]
    top_aspect_keywords = {
        a: [k for k, _ in c.most_common(TOP_N_ASPECT_KEYWORDS)]
        for a, c in aspect_keywords.items()
    }
    return top_keywords, top_aspect_keywords

# -----------------------
# feature keywords (✅ 향/냄새 포함하도록 추가)
# -----------------------
FEATURE_SEEDS = {
    "흡수/샘": {"흡수","흡수력","샘","새","역류","뭉침","양많","오버나이트"},
    "민감/자극": {"자극","가려","가렵","따가","트러블","민감","발진","간지","피부","순하","저자극"},
    "두께/슬림": {"두께","얇","슬림","도톰","두껍","부피","티","휴대"},
    "착용감": {"편안","편하","편해","쾌적","부드럽","촉감","보송","산뜻","밀착","이물감","답답","말림","쓸림","구겨"},
    "접착/고정": {"접착","고정","고정력","밀림","떨어","날개"},
    "사이즈/핏": {"사이즈","길이","너비","폭","대형","중형","소형","롱","핏","커버"},
    "향/냄새": {"향","냄새","무향","향료","향이","비린","비릿","잡냄새","체취","향기","향긋"},  # ✅ 추가
}

def build_feature_keywords_weighted(keyphrases: List[str], phrase_score_map: Dict[str, float], topn=12):
    feat_scores = {a: Counter() for a in FEATURE_SEEDS.keys()}
    for ph in keyphrases:
        score = float(phrase_score_map.get(ph, 0.0))
        if score <= 0:
            continue
        toks = phrase_to_keywords(ph)
        if not toks:
            continue
        for tok in toks:
            for aspect, seeds in FEATURE_SEEDS.items():
                # seed 부분문자열 매칭 유지
                if any(s in tok for s in seeds):
                    feat_scores[aspect][tok] += score

    feature_keywords = {a: [k for k, _ in c.most_common(topn)] for a, c in feat_scores.items()}
    feature_score_map = {a: dict(c) for a, c in feat_scores.items()}
    return feature_keywords, feature_score_map

# =========================
# schema helpers
# =========================
def keywords_table_has_posneg(conn) -> bool:
    """product_review_keywords에 pos/neg 관련 컬럼이 실제로 존재하면 True"""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*)
            FROM information_schema.columns
            WHERE table_name='product_review_keywords'
              AND column_name IN ('pos_keyword','neg_keyword','pos_keyword_num','neg_keyword_num');
        """)
        cnt = cur.fetchone()[0]
    return cnt == 4

# =========================
# DB upsert
# =========================
def upsert_product_review_summary(conn, ext_id: str, product_id: int, pos_ratio: float, neg_ratio: float,
                                 pos_keywords: list, neg_keywords: list):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO product_review_summary(
              product_id, external_product_id,
              pos_ratio, neg_ratio,
              pos_keyword, neg_keyword,
              pos_keyword_num, neg_keyword_num
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (external_product_id) DO UPDATE SET
              product_id = EXCLUDED.product_id,
              pos_ratio = EXCLUDED.pos_ratio,
              neg_ratio = EXCLUDED.neg_ratio,
              pos_keyword = EXCLUDED.pos_keyword,
              neg_keyword = EXCLUDED.neg_keyword,
              pos_keyword_num = EXCLUDED.pos_keyword_num,
              neg_keyword_num = EXCLUDED.neg_keyword_num
            """,
            (
                int(product_id),
                ext_id,
                float(pos_ratio),
                float(neg_ratio),
                json.dumps(pos_keywords, ensure_ascii=False),
                json.dumps(neg_keywords, ensure_ascii=False),
                int(len(pos_keywords)),
                int(len(neg_keywords)),
            ),
        )
    conn.commit()

def upsert_product_review_keywords(conn, ext_id: str, product_id: int,
                                  top_keywords: list,
                                  aspect_keywords: dict,
                                  source_keyphrases: list,
                                  feature_keywords: dict,
                                  feature_scores: dict,
                                  pos_keywords: list,
                                  neg_keywords: list,
                                  also_posneg: bool):
    if also_posneg:
        sql = """
            INSERT INTO product_review_keywords(
              product_id, external_product_id,
              top_keywords, aspect_keywords, source_keyphrases,
              feature_keywords, feature_scores,
              pos_keyword, neg_keyword, pos_keyword_num, neg_keyword_num
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (external_product_id) DO UPDATE SET
              product_id = EXCLUDED.product_id,
              top_keywords = EXCLUDED.top_keywords,
              aspect_keywords = EXCLUDED.aspect_keywords,
              source_keyphrases = EXCLUDED.source_keyphrases,
              feature_keywords = EXCLUDED.feature_keywords,
              feature_scores = EXCLUDED.feature_scores,
              pos_keyword = EXCLUDED.pos_keyword,
              neg_keyword = EXCLUDED.neg_keyword,
              pos_keyword_num = EXCLUDED.pos_keyword_num,
              neg_keyword_num = EXCLUDED.neg_keyword_num
        """
        params = (
            int(product_id),
            ext_id,
            json.dumps(top_keywords, ensure_ascii=False),
            json.dumps(aspect_keywords, ensure_ascii=False),
            json.dumps(source_keyphrases, ensure_ascii=False),
            json.dumps(feature_keywords, ensure_ascii=False),
            json.dumps(feature_scores, ensure_ascii=False),
            json.dumps(pos_keywords, ensure_ascii=False),
            json.dumps(neg_keywords, ensure_ascii=False),
            int(len(pos_keywords)),
            int(len(neg_keywords)),
        )
    else:
        sql = """
            INSERT INTO product_review_keywords(
              product_id, external_product_id,
              top_keywords, aspect_keywords, source_keyphrases,
              feature_keywords, feature_scores
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (external_product_id) DO UPDATE SET
              product_id = EXCLUDED.product_id,
              top_keywords = EXCLUDED.top_keywords,
              aspect_keywords = EXCLUDED.aspect_keywords,
              source_keyphrases = EXCLUDED.source_keyphrases,
              feature_keywords = EXCLUDED.feature_keywords,
              feature_scores = EXCLUDED.feature_scores
        """
        params = (
            int(product_id),
            ext_id,
            json.dumps(top_keywords, ensure_ascii=False),
            json.dumps(aspect_keywords, ensure_ascii=False),
            json.dumps(source_keyphrases, ensure_ascii=False),
            json.dumps(feature_keywords, ensure_ascii=False),
            json.dumps(feature_scores, ensure_ascii=False),
        )

    with conn.cursor() as cur:
        cur.execute(sql, params)
    conn.commit()

# =========================
# Per-product pipeline
# =========================
def extract_posneg_keywords(filtered_pairs: List[Tuple[str, float]]) -> Tuple[List[str], List[str]]:
    # 긍/부정 리뷰 분리 후 각각 키프레이즈 → 키워드
    pos_reviews = [t for (t, sc) in filtered_pairs if pd.notna(sc) and float(sc) >= 4]
    neg_reviews = [t for (t, sc) in filtered_pairs if pd.notna(sc) and float(sc) <= 2]

    pos_phr, pos_score_map = extract_keyphrases_all(pos_reviews, top_n=30) if pos_reviews else ([], {})
    neg_phr, neg_score_map = extract_keyphrases_all(neg_reviews, top_n=30) if neg_reviews else ([], {})

    pos_kw, _ = build_keywords_from_keyphrases_weighted(pos_phr, pos_score_map) if pos_phr else ([], {})
    neg_kw, _ = build_keywords_from_keyphrases_weighted(neg_phr, neg_score_map) if neg_phr else ([], {})

    return pos_kw[:30], neg_kw[:30]

def run_batch(limit_products: Optional[int] = None):
    with db_conn() as conn:
        also_posneg_in_keywords = keywords_table_has_posneg(conn)

        # 1) 리뷰에 존재하는 product_id 목록
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT r.product_id
                FROM review r
                WHERE r.product_id IS NOT NULL
                ORDER BY r.product_id
            """)
            product_ids = [r[0] for r in cur.fetchall()]

        if limit_products:
            product_ids = product_ids[:limit_products]

        for product_id in product_ids:
            # 2) product에서 external_product_id 확보
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT external_product_id FROM product WHERE id=%s",
                    (product_id,),
                )
                pr = cur.fetchone()

            if not pr or not pr[0]:
                continue

            ext_id = pr[0]

            # 3) 리뷰 로드
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT content, rating
                    FROM review
                    WHERE product_id = %s
                    """,
                    (product_id,),
                )
                rows = cur.fetchall()

            if not rows:
                continue

            filtered_pairs = []
            for content, score in rows:
                t = clean_text(content) if content is not None else ""
                if len(t) >= 5:
                    filtered_pairs.append((t, score))

            reviews = [t for t, _ in filtered_pairs]
            scores = [sc for _, sc in filtered_pairs]

            if len(reviews) < MIN_REVIEWS_FOR_KEYWORDS:
                continue

            pos_ratio, neu_ratio, neg_ratio = sentiment_ratio(scores)

            # 전체 키워드/피처
            keyphrases_all, phrase_score_map_all = extract_keyphrases_all(reviews, top_n=TOP_N_KEYPHRASES)
            top_keywords, aspect_keywords = build_keywords_from_keyphrases_weighted(
                keyphrases_all, phrase_score_map_all, alpha_len_norm=0.7
            )

            feature_keywords, feature_scores = build_feature_keywords_weighted(
                keyphrases_all, phrase_score_map_all, topn=12
            )

            pos_kw, neg_kw = extract_posneg_keywords(filtered_pairs)

            upsert_product_review_summary(
                conn,
                ext_id=ext_id,
                product_id=int(product_id),
                pos_ratio=pos_ratio,
                neg_ratio=neg_ratio,
                pos_keywords=pos_kw,
                neg_keywords=neg_kw,
            )

            upsert_product_review_keywords(
                conn,
                ext_id=ext_id,
                product_id=int(product_id),
                top_keywords=top_keywords[:30],
                aspect_keywords=aspect_keywords,          # ✅ 여기에 향/냄새 키도 들어올 수 있음
                source_keyphrases=keyphrases_all[:50],
                feature_keywords=feature_keywords,        # ✅ 향/냄새 포함
                feature_scores=feature_scores,            # ✅ 향/냄새 포함
                pos_keywords=pos_kw,
                neg_keywords=neg_kw,
                also_posneg=also_posneg_in_keywords,
            )

            print("saved:", ext_id, "product_id:", product_id, "reviews:", len(reviews),
                  "pos_kw:", len(pos_kw), "neg_kw:", len(neg_kw))


if __name__ == "__main__":
    run_batch(limit_products=940)
