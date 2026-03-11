from dataclasses import dataclass
from typing import Optional

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from heart_risk.data.preprocess import PreprocessConfig


@dataclass
class ArtifactConfig:
    model_path: str = "artifacts/model.cbm"
    preprocessor_path: str = "artifacts/preprocessor.joblib"


class Predictor:
    def __init__(self, cfg: Optional[ArtifactConfig] = None):
        self.cfg = cfg or ArtifactConfig()
        self.model = CatBoostClassifier()
        self.preprocessor = None

    def load(self) -> "Predictor":
        self.model.load_model(self.cfg.model_path)
        self.preprocessor = joblib.load(self.cfg.preprocessor_path)
        return self

    def predict_proba(self, df: pd.DataFrame):
        if self.preprocessor is None:
            raise RuntimeError("Predictor is not loaded. Call load().")
        X, _, ids = self.preprocessor.split_X_y_id(df)
        X_mat = self.preprocessor.transform(df)
        proba = self.model.predict_proba(X_mat)[:, 1]
        return ids, proba