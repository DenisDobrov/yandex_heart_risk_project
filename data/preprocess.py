from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class PreprocessConfig:
    target_col: str = "Heart Attack Risk (Binary)"
    id_col: str = "id"
    drop_cols: Tuple[str, ...] = ("Unnamed: 0",)

    # признаки с пропусками (по вашему выводу)
    binary_with_nans: Tuple[str, ...] = (
        "Diabetes",
        "Family History",
        "Smoking",
        "Obesity",
        "Alcohol Consumption",
        "Previous Heart Problems",
        "Medication Use",
        "Stress Level",
        "Physical Activity Days Per Week",
    )
    categorical_cols: Tuple[str, ...] = ("Gender",)


class DataPreprocessor:
    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_: Optional[List[str]] = None

    def _build_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        # Все колонки кроме drop/id/target
        excluded = set(self.config.drop_cols) | {self.config.id_col, self.config.target_col}
        cols = [c for c in X.columns if c not in excluded]

        cat_cols = [c for c in cols if c in self.config.categorical_cols]
        bin_cols = [c for c in cols if c in self.config.binary_with_nans]
        num_cols = [c for c in cols if (c not in cat_cols and c not in bin_cols)]

        binary_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ])

        categorical_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])

        numeric_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])

        ct = ColumnTransformer(
            transformers=[
                ("bin", binary_pipe, bin_cols),
                ("cat", categorical_pipe, cat_cols),
                ("num", numeric_pipe, num_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return ct

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        df = df.copy()
        df = df.drop(columns=[c for c in self.config.drop_cols if c in df.columns], errors="ignore")
        self.pipeline = self._build_pipeline(df)
        X = df
        self.pipeline.fit(X)
        self.feature_names_ = list(self.pipeline.get_feature_names_out())
        return self

    def transform(self, df: pd.DataFrame):
        if self.pipeline is None:
            raise RuntimeError("Preprocessor is not fitted. Call fit() first.")
        df = df.copy()
        df = df.drop(columns=[c for c in self.config.drop_cols if c in df.columns], errors="ignore")
        return self.pipeline.transform(df)

    def split_X_y_id(self, df: pd.DataFrame):
        df = df.copy()
        df = df.drop(columns=[c for c in self.config.drop_cols if c in df.columns], errors="ignore")

        ids = df[self.config.id_col].astype(int) if self.config.id_col in df.columns else None
        y = df[self.config.target_col].astype(int) if self.config.target_col in df.columns else None

        drop_for_X = [c for c in [self.config.id_col, self.config.target_col] if c in df.columns]
        X = df.drop(columns=drop_for_X, errors="ignore")
        return X, y, ids