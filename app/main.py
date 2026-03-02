import os
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
from psycopg.rows import dict_row

from .db import get_conn
from .reco_model import RecoModel
from .sql import CANDIDATE_SQL
from .routers.keyword_router import router as keyword_router   
from .routers.user_recommend_router import router as user_recommend_router 
from .routers.review_keyword_router import router as review_keyword_router
from .routers.pos_neg_keyword_router import router as pos_neg_keyword_router
from .routers.trend_summary_router import router as trend_summary_router
from .routers.sentiment_ratio_router import router as sentiment_ratio_router
from .routers.keyword_update_router import router as keyword_update_router

load_dotenv()

MODEL_PATH = os.getenv("RECO_MODEL_PATH", "./models/rec_lgbm_rank.txt")
TOPK = int(os.getenv("RECO_TOPK", "20"))
CANDIDATES_PER_CAT = int(os.getenv("RECO_CANDIDATES_PER_CAT", "200"))

app = FastAPI(title="NUDA ML API")
app.include_router(keyword_router)
app.include_router(user_recommend_router)
app.include_router(review_keyword_router)
app.include_router(pos_neg_keyword_router)
app.include_router(trend_summary_router)
app.include_router(sentiment_ratio_router)
app.include_router(keyword_update_router)

model = RecoModel(MODEL_PATH)

@app.get("/health",summary="헬스체크용 API")
def health():
    return {"ok": True, "model_loaded": True}

