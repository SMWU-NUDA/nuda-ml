# app/reco_model.py

import lightgbm as lgb
import pandas as pd


class RecoModel:

    def __init__(self, model_path: str):

        # 모델 로드
        self.model = lgb.Booster(model_file=model_path)

        # ✅ 모델이 사용하는 feature 이름 저장
        self.feature_names = self.model.feature_name()


    def score(self, df: pd.DataFrame):

        # 모델 feature 순서 맞추기
        X = df[self.feature_names].copy()

        # dtype 안전 변환
        for c in self.feature_names:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        X = X.fillna(0)

        # 예측
        return self.model.predict(X)
