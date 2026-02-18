from typing import Optional
import os, re, json, random, unicodedata, math
from collections import Counter, defaultdict

import pandas as pd
import psycopg
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# =======================
# Env
# =======================
load_dotenv()

# =======================
# Model (CPU 고정)
# =======================
MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
st_model = SentenceTransformer(MODEL_NAME, device="cpu")
kw_model = KeyBERT(model=st_model)

# =======================
# Settings
# =======================
MIN_REVIEWS_FOR_SUMMARY = 10

CHUNK_SIZE = 200
MAX_REVIEW_CHARS = 220
MAX_DOC_CHARS = 12000

TOP_N_KEYPHRASES = 50
CANDIDATE_MULTIPLIER = 4

TOP_N_KEYWORDS = 30
TOP_N_ASPECT_KEYWORDS = 12

STOPWORDS = {
    "배송", "포장", "가격", "쿠폰", "사은품", "할인", "구매", "주문", "재구매", "추천", "만족",
    "제품", "상품", "사용", "이번", "항상", "진짜", "너무", "그냥", "완전", "약간", "정말",
    "올리브영", "올영", "마트", "행사", "세일", "증정", "무료", "구입",
    "기획", "패키지", "구성", "브랜드", "쓰는", "세면", "생리대"
}

# -----------------------
# 1) 전처리
# -----------------------
CTRL_RE = re.compile(
    r"[\u0000-\u001F\u007F]"
    r"|[\u200B-\u200F\u202A-\u202E]"
    r"|[\u2028\u2029]"
)
VS_RE = re.compile(r"[\uFE0E\uFE0F]")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "\u2B00-\u2BFF"
    "]+",
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
# 2) Aspect Mapping Rules (✅ 향/냄새 포함)
# -----------------------
ASPECT_RULES = {
    "흡수/샘": [
        "흡수", "흡수력", "샘", "새", "샜", "새요", "새는", "새지", "샘방지",
        "양많", "오버나이트", "밤", "많은날", "역류", "뭉침"
    ],
    "민감/자극": [
        "자극", "가려", "가렵", "따가", "따갑", "트러블", "민감", "발진", "간지", "피부",
        "가려움", "가렵지", "자극없", "순하", "저자극", "편안", "부드럽"
    ],
    "향/냄새": [  # ✅ 여기
        "향", "냄새", "무향", "향료", "향이", "냄새가", "비린", "잡냄새", "향기", "향긋"
    ],
    "두께/슬림": [
        "얇", "얇아", "슬림", "두께", "두껍", "도톰", "두툼", "도톰하",
        "티안", "티나", "부피", "휴대", "가볍"
    ],
    "착용감": [
        "편안", "편하", "편해", "쾌적", "가볍", "부드럽", "촉감", "보송", "산뜻",
        "불편", "이물감", "착용", "답답", "움직", "말림", "구겨", "쓸림", "쓸리", "밀착"
    ],
    "접착/고정": [
        "접착", "고정", "고정력", "잘붙", "붙어", "떨어", "안떨어", "밀림", "밀려",
        "움직임", "날개", "뜯", "찢", "접힘"
    ],
    "사이즈/핏": [
        "대형", "중형", "소형", "롱", "팬티라이너", "길이", "사이즈", "핏", "너비", "폭",
        "커버", "면적", "날개길이", "크기"
    ],
}

def map_phrase_to_aspect(phrase: str) -> Optional[str]:
    p = phrase.replace(" ", "")
    for aspect, kws in ASPECT_RULES.items():
        for k in kws:
            if k in p:
                return aspect
    return None

# -----------------------
# 3) 감성 비율 (별점 기반)
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
# 4) Keyphrase 필터
# -----------------------
BAD_PATTERNS = re.compile(
    r"(\d{1,2}월|\d{1,2}일|\d{4}|\d+%|원|까지|담달|행사|세일|올영|올리브영|마트|쿠폰|가격|포장|배송)"
)
JOSA_ENDINGS = (
    "이다", "네요", "해요", "합니다", "같아요", "있어요", "없어요", "좋아요", "싫어요",
    "했어요", "되네요", "되요", "에요", "입니다"
)

def is_good_keyphrase(ph: str) -> bool:
    ph = ph.strip()
    if len(ph) < 2:
        return False
    if len(ph) > 20:
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

# -----------------------
# 5) 리뷰 전체 커버 + KeyBERT 누적 (✅ phrase 점수까지 리턴)
# -----------------------
def chunk_reviews(reviews: list[str], chunk_size: int) -> list[list[str]]:
    return [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]

def extract_keyphrases_all(reviews: list[str], top_n=TOP_N_KEYPHRASES):
    """
    Returns:
      - keyphrases: list[str]           (상위 phrase만)
      - phrase_scores: dict[str, float] (phrase -> 누적 점수)
    """
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
            diversity=0.65
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
# 6) Keyphrase -> Keyword 정규화
# -----------------------
TOKEN_BAD = re.compile(r"(\d|%|원|까지|담달|행사|세일|올영|올리브영|배송|포장|쿠폰|가격)")

KOREAN_JOSA_END = (
    "은","는","이","가","을","를","에","에서","에게","한테","로","으로",
    "와","과","랑","하고","도","만","까지","부터","마다","이나","나",
    "처럼","같이","보다","의","께","뿐","조차","마저"
)

COLLOQUIAL_STOP = {
    "좋더라구요","좋아요","좋네","좋음","짱","짱짱","대박","추천","만족",
    "쓰는중","쓰는중입니당","써요","사용해요","사용중","계속","항상","매번",
    "그냥","진짜","너무","완전","약간","정말","괜찮아요","괜찮음","괜찮게","않아용",
    "입니당","이에요","예요","네요","해요","했어요","되요","돼요",
    "것","거","부분","이것","저것"
}

ENDING_TRIMS = (
    "입니다","이에요","예요","네요","해요","했어요","했네","했음","합니다",
    "같아요","있어요","없어요","되요","돼요","임","요"
)

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

def phrase_to_keywords(phrase: str) -> list[str]:
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

# -----------------------
# 7) token-level weighted scoring (✅ 향/냄새도 aspect로 들어옴)
# -----------------------
def build_keywords_from_keyphrases_weighted(
    keyphrases: list[str],
    phrase_score_map: dict[str, float],
    alpha_len_norm: float = 0.7,
    distribute: str = "equal"
):
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

        if distribute == "equal":
            per_tok = ph_score * len_norm
            for t in toks:
                keyword_scores[t] += per_tok
                if aspect:
                    aspect_keywords[aspect][t] += per_tok
        elif distribute == "full":
            per_tok = ph_score
            for t in toks:
                keyword_scores[t] += per_tok
                if aspect:
                    aspect_keywords[aspect][t] += per_tok
        else:
            raise ValueError("distribute must be 'equal' or 'full'")

    top_keywords = [k for k, _ in keyword_scores.most_common(TOP_N_KEYWORDS)]
    top_aspect_keywords = {
        a: [k for k, _ in c.most_common(TOP_N_ASPECT_KEYWORDS)]
        for a, c in aspect_keywords.items()
    }
    return top_keywords, top_aspect_keywords

# -----------------------
# 8) 추천용 Feature 키워드 (✅ 향/냄새 포함)
# -----------------------
FEATURE_SEEDS = {
    "흡수/샘": {"흡수", "흡수력", "샘", "새", "역류", "뭉침", "양많", "오버나이트"},
    "민감/자극": {"자극", "가려", "가렵", "따가", "트러블", "민감", "발진", "간지", "피부", "순하", "저자극"},
    "두께/슬림": {"두께", "얇", "슬림", "도톰", "두껍", "부피", "티", "휴대"},
    "착용감": {"편안", "편하", "편해", "쾌적", "부드럽", "촉감", "보송", "산뜻", "밀착", "이물감", "답답", "말림", "쓸림", "구겨"},
    "접착/고정": {"접착", "고정", "고정력", "밀림", "떨어", "날개"},
    "사이즈/핏": {"사이즈", "길이", "너비", "폭", "대형", "중형", "소형", "롱", "핏", "커버"},
    "향/냄새": {"향", "냄새", "무향", "향료", "향이", "비린", "잡냄새", "향기", "향긋"},  # ✅ 여기
}

FEATURE_STOP = {
    "제일","자주","유용","리뷰","애용","믿고","너무","너무너무","무난","괜찮",
    "제품","상품","구매","주문","재구매","추천","만족","할인","행사","세일",
    "사용","쓰","사용감","사용해","사용하","사용중"
}

def token_is_feature_like(tok: str) -> bool:
    if tok in FEATURE_STOP:
        return False
    if len(tok) < 2 or len(tok) > 8:
        return False
    if tok.endswith(("입니당","더라구","더라","하구","했고","했어","했어요","해요","네요","요")):
        return False
    return True

def build_feature_keywords_weighted(keyphrases: list[str], phrase_score_map: dict[str, float], topn=12):
    """
    Returns:
      feature_keywords: {aspect: [kw1, kw2, ...]}
      feature_scores:   {aspect: {kw: score}}
    """
    feat_scores = {a: Counter() for a in FEATURE_SEEDS.keys()}

    for ph in keyphrases:
        score = float(phrase_score_map.get(ph, 0.0))
        if score <= 0:
            continue

        toks = phrase_to_keywords(ph)
        if not toks:
            continue

        for tok in toks:
            if not token_is_feature_like(tok):
                continue

            for aspect, seeds in FEATURE_SEEDS.items():
                if any(s in tok for s in seeds):
                    feat_scores[aspect][tok] += score

    feature_keywords = {a: [k for k, _ in c.most_common(topn)] for a, c in feat_scores.items()}
    feature_score_map = {a: dict(c) for a, c in feat_scores.items()}
    return feature_keywords, feature_score_map

# -----------------------
# 9) 3줄 요약
# -----------------------
def build_3line_summary(review_count, pos, neg, top_aspects, aspect_to_phrases):
    line1 = f"리뷰 {review_count}개 기준, 긍정 비율은 {pos*100:.0f}%, 부정 비율은 {neg*100:.0f}%이다."

    if top_aspects:
        a1 = top_aspects[0]
        p1 = aspect_to_phrases.get(a1, [])
        key_part = ", ".join(p1[:3]) if p1 else a1
        line2 = f"가장 많이 언급된 포인트는 [{a1}]이며, 주요 키워드는 {key_part}이다."
    else:
        line2 = "리뷰에서 특정 포인트가 반복적으로 두드러지지는 않지만 전반 의견을 기반으로 요약할 수 있다."

    if len(top_aspects) >= 2:
        a2 = top_aspects[1]
        p2 = aspect_to_phrases.get(a2, [])
        key_part2 = ", ".join(p2[:2]) if p2 else a2
    else:
        a2, key_part2 = None, None

    if neg < 0.10:
        line3 = f"주의 의견은 상대적으로 적으며{'' if not a2 else f', [{a2}] 관련으로 {key_part2} 같은 언급이 일부 있다'}."
    else:
        line3 = f"불만은 주로{'' if not a2 else f' [{a2}]에서'} 나타나며, {key_part2 if key_part2 else '사용감 차이에 따른 의견'}이 보인다."

    return "\n".join([line1, line2, line3])

# -----------------------
# 10) 제품 1개 요약
# -----------------------
def summarize_one_product(rows: list[tuple]):
    external_product_id = rows[0][0]
    category_code = rows[0][1]

    filtered_pairs = []
    for _, _, content, score in rows:
        t = clean_text(content) if content is not None else ""
        if len(t) >= 5:
            filtered_pairs.append((t, score))

    reviews = [t for t, _ in filtered_pairs]
    scores = [sc for _, sc in filtered_pairs]
    review_count = len(reviews)

    if review_count < MIN_REVIEWS_FOR_SUMMARY:
        return {
            "external_product_id": external_product_id,
            "category_code": category_code,
            "review_count": review_count,
            "pos_ratio": None,
            "neg_ratio": None,
            "top_aspects": {},
            "top_keyphrases": [],
            "summary_3lines": None,
            "top_keywords": [],
            "aspect_keywords": {},
            "feature_keywords": {},
            "feature_scores": {},
        }

    pos, neu, neg = sentiment_ratio(scores)

    # ✅ phrase + 점수
    keyphrases, phrase_score_map = extract_keyphrases_all(reviews, top_n=TOP_N_KEYPHRASES)

    # phrase -> aspect
    aspect_to_phrases = defaultdict(list)
    for ph in keyphrases:
        aspect = map_phrase_to_aspect(ph)
        if aspect:
            aspect_to_phrases[aspect].append(ph)

    aspect_rank = sorted(
        aspect_to_phrases.keys(),
        key=lambda a: len(aspect_to_phrases[a]),
        reverse=True
    )
    top_aspects = aspect_rank[:3]

    summary = build_3line_summary(
        review_count=review_count,
        pos=pos,
        neg=neg,
        top_aspects=top_aspects,
        aspect_to_phrases=aspect_to_phrases
    )

    # ✅ weighted keyword
    top_keywords, aspect_keywords = build_keywords_from_keyphrases_weighted(
        keyphrases=keyphrases,
        phrase_score_map=phrase_score_map,
        alpha_len_norm=0.7,
        distribute="equal"
    )

    # ✅ 추천용 feature (향/냄새 포함)
    feature_keywords, feature_scores = build_feature_keywords_weighted(
        keyphrases=keyphrases,
        phrase_score_map=phrase_score_map,
        topn=12
    )

    return {
        "external_product_id": external_product_id,
        "category_code": category_code,
        "review_count": review_count,
        "pos_ratio": pos,
        "neg_ratio": neg,
        "top_aspects": {a: aspect_to_phrases[a][:5] for a in top_aspects},
        "top_keyphrases": keyphrases[:20],
        "summary_3lines": summary,
        "top_keywords": top_keywords,
        "aspect_keywords": aspect_keywords,          # ✅ 여기에 "향/냄새" 키가 생김
        "feature_keywords": feature_keywords,        # ✅ 여기에 "향/냄새" 키가 생김
        "feature_scores": feature_scores,
    }

# -----------------------
# 11) 테이블 보장
# -----------------------
def ensure_summary_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS product_review_summary (
              external_product_id TEXT PRIMARY KEY,
              category_code TEXT,
              review_count INTEGER NOT NULL,
              pos_ratio DOUBLE PRECISION,
              neg_ratio DOUBLE PRECISION,
              top_aspects JSONB,
              top_keyphrases JSONB,
              summary_3lines TEXT,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
    conn.commit()

def ensure_keywords_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS product_review_keywords (
              external_product_id TEXT PRIMARY KEY,
              category_code TEXT,
              review_count INTEGER NOT NULL,
              top_keywords JSONB,
              aspect_keywords JSONB,
              feature_keywords JSONB,
              feature_scores JSONB,
              source_keyphrases JSONB,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
    conn.commit()

# -----------------------
# 12) DB 배치 실행
# -----------------------
def run_batch(limit_products=None):
    with psycopg.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    ) as conn:

        ensure_summary_table(conn)
        ensure_keywords_table(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT external_product_id FROM review")
            product_ids = [r[0] for r in cur.fetchall()]

        if limit_products:
            product_ids = product_ids[:limit_products]

        for pid in product_ids:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT external_product_id, category_code, review_content, review_score
                    FROM review
                    WHERE external_product_id = %s
                """, (pid,))
                rows = cur.fetchall()

            if not rows:
                continue

            result = summarize_one_product(rows)

            # ✅ 요약 테이블
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO product_review_summary(
                        external_product_id, category_code, review_count,
                        pos_ratio, neg_ratio, top_aspects, top_keyphrases, summary_3lines
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (external_product_id) DO UPDATE SET
                      category_code = EXCLUDED.category_code,
                      review_count = EXCLUDED.review_count,
                      pos_ratio = EXCLUDED.pos_ratio,
                      neg_ratio = EXCLUDED.neg_ratio,
                      top_aspects = EXCLUDED.top_aspects,
                      top_keyphrases = EXCLUDED.top_keyphrases,
                      summary_3lines = EXCLUDED.summary_3lines,
                      updated_at = now()
                """, (
                    result["external_product_id"],
                    result["category_code"],
                    result["review_count"],
                    result["pos_ratio"],
                    result["neg_ratio"],
                    json.dumps(result["top_aspects"], ensure_ascii=False),
                    json.dumps(result["top_keyphrases"], ensure_ascii=False),
                    result["summary_3lines"],
                ))

            # ✅ 키워드 테이블
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO product_review_keywords(
                        external_product_id, category_code, review_count,
                        top_keywords, aspect_keywords,
                        feature_keywords, feature_scores,
                        source_keyphrases
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (external_product_id) DO UPDATE SET
                      category_code = EXCLUDED.category_code,
                      review_count = EXCLUDED.review_count,
                      top_keywords = EXCLUDED.top_keywords,
                      aspect_keywords = EXCLUDED.aspect_keywords,
                      feature_keywords = EXCLUDED.feature_keywords,
                      feature_scores = EXCLUDED.feature_scores,
                      source_keyphrases = EXCLUDED.source_keyphrases,
                      updated_at = now()
                """, (
                    result["external_product_id"],
                    result["category_code"],
                    result["review_count"],
                    json.dumps(result["top_keywords"], ensure_ascii=False),
                    json.dumps(result["aspect_keywords"], ensure_ascii=False),
                    json.dumps(result["feature_keywords"], ensure_ascii=False),
                    json.dumps(result["feature_scores"], ensure_ascii=False),
                    json.dumps(result["top_keyphrases"], ensure_ascii=False),
                ))

            conn.commit()
            print("saved:", pid, "reviews:", result["review_count"])

if __name__ == "__main__":
    run_batch(limit_products=928)
