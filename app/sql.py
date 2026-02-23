EMB_CANDIDATE_IDS_SQL = """
SELECT p.id AS product_id
FROM ml_product_embedding e
JOIN product p ON p.id = e.product_id
WHERE p.external_product_id NOT LIKE 'INTERNAL-%%'
ORDER BY e.embedding <=> (%s::text::vector)
LIMIT %s
"""

CANDIDATE_SQL = """
WITH member_pref AS (
  SELECT
    member_id,
    p_sensitivity_level::float8 AS p_sensitivity_level,
    p_scent_level::float8       AS p_scent_level,
    p_absorbency_level::float8  AS p_absorbency_level,
    p_adhesion_level::float8    AS p_adhesion_level,
    p_safety_level::float8      AS p_safety_level
  FROM rec_member_pref
  WHERE member_id = %s
),

candidates AS (
  SELECT
    p.id AS product_id,
    p.external_product_id,

    -- feature_all이 없어도 후보에서 안 빠지게
    COALESCE(f.sensitivity_level, 3)::float8 AS sensitivity_level,
    COALESCE(f.scent_level, 3)::float8       AS scent_level,
    COALESCE(f.absorbency_level, 3)::float8  AS absorbency_level,
    COALESCE(f.adhesion_level, 3)::float8    AS adhesion_level,
    COALESCE(f.safety_level, 3)::float8      AS safety_level,

    -- 신뢰도(리뷰 수)
    COALESCE(rs.review_count, 0)::float8     AS review_count

  FROM product p
  LEFT JOIN v_product_feature_all f
    ON f.product_id = p.id
  LEFT JOIN v_product_review_stats rs
    ON rs.product_id = p.id
  WHERE p.external_product_id NOT LIKE 'INTERNAL-%%'
  ORDER BY (f.product_id IS NULL), random()
  LIMIT %s
)

SELECT
  c.product_id,
  c.external_product_id,

  -- p_* 5개 (pref 없으면 3)
  COALESCE(mp.p_sensitivity_level, 3)::float8 AS p_sensitivity_level,
  COALESCE(mp.p_scent_level, 3)::float8       AS p_scent_level,
  COALESCE(mp.p_absorbency_level, 3)::float8  AS p_absorbency_level,
  COALESCE(mp.p_adhesion_level, 3)::float8    AS p_adhesion_level,
  COALESCE(mp.p_safety_level, 3)::float8      AS p_safety_level,

  -- 상품 레벨
  c.sensitivity_level,
  c.scent_level,
  c.absorbency_level,
  c.adhesion_level,
  c.safety_level,

  -- 신뢰도 피처(모델 입력)
  c.review_count,

  -- 거리
  ABS(COALESCE(mp.p_sensitivity_level, 3) - c.sensitivity_level)::float8 AS d_sensitivity,
  ABS(COALESCE(mp.p_scent_level, 3)       - c.scent_level)::float8       AS d_scent,
  ABS(COALESCE(mp.p_absorbency_level, 3)  - c.absorbency_level)::float8  AS d_absorbency,
  ABS(COALESCE(mp.p_adhesion_level, 3)    - c.adhesion_level)::float8    AS d_adhesion,
  ABS(COALESCE(mp.p_safety_level, 3)      - c.safety_level)::float8      AS d_safety

FROM candidates c
LEFT JOIN member_pref mp ON true;
"""

CANDIDATE_BY_IDS_SQL = """
WITH member_pref AS (
  SELECT
    member_id,
    p_sensitivity_level::float8 AS p_sensitivity_level,
    p_scent_level::float8       AS p_scent_level,
    p_absorbency_level::float8  AS p_absorbency_level,
    p_adhesion_level::float8    AS p_adhesion_level,
    p_safety_level::float8      AS p_safety_level
  FROM rec_member_pref
  WHERE member_id = %s
),

candidates AS (
  SELECT
    p.id AS product_id,
    p.external_product_id,

    COALESCE(f.sensitivity_level, 3)::float8 AS sensitivity_level,
    COALESCE(f.scent_level, 3)::float8       AS scent_level,
    COALESCE(f.absorbency_level, 3)::float8  AS absorbency_level,
    COALESCE(f.adhesion_level, 3)::float8    AS adhesion_level,
    COALESCE(f.safety_level, 3)::float8      AS safety_level,

    COALESCE(rs.review_count, 0)::float8     AS review_count
  FROM product p
  LEFT JOIN v_product_feature_all f
    ON f.product_id = p.id
  LEFT JOIN v_product_review_stats rs
    ON rs.product_id = p.id
  WHERE p.id = ANY(%s)   -- 핵심: 후보 id list만 대상
)

SELECT
  c.product_id,
  c.external_product_id,

  COALESCE(mp.p_sensitivity_level, 3)::float8 AS p_sensitivity_level,
  COALESCE(mp.p_scent_level, 3)::float8       AS p_scent_level,
  COALESCE(mp.p_absorbency_level, 3)::float8  AS p_absorbency_level,
  COALESCE(mp.p_adhesion_level, 3)::float8    AS p_adhesion_level,
  COALESCE(mp.p_safety_level, 3)::float8      AS p_safety_level,

  c.sensitivity_level,
  c.scent_level,
  c.absorbency_level,
  c.adhesion_level,
  c.safety_level,

  c.review_count,

  ABS(COALESCE(mp.p_sensitivity_level, 3) - c.sensitivity_level)::float8 AS d_sensitivity,
  ABS(COALESCE(mp.p_scent_level, 3)       - c.scent_level)::float8       AS d_scent,
  ABS(COALESCE(mp.p_absorbency_level, 3)  - c.absorbency_level)::float8  AS d_absorbency,
  ABS(COALESCE(mp.p_adhesion_level, 3)    - c.adhesion_level)::float8    AS d_adhesion,
  ABS(COALESCE(mp.p_safety_level, 3)      - c.safety_level)::float8      AS d_safety

FROM candidates c
LEFT JOIN member_pref mp ON true;
"""