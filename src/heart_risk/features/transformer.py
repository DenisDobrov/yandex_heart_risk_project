import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class HeartFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.target_col = 'Heart Attack Risk (Binary)'
        self.drop_cols = ['Unnamed: 0', 'id']

    def fit(self, X, y=None):
        # Здесь можно вычислить медианы или моды, если нужно
        self.medians = X.median(numeric_only=True)
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # 1. Удаляем ненужные колонки
        cols_to_drop = [c for c in self.drop_cols if c in X_copy.columns]
        X_copy = X_copy.drop(columns=cols_to_drop)
        
        # 2. Обработка пола (Gender)
        # CatBoost умеет работать с категориями, но для надежности 
        # приведем к типу category или закодируем
        if 'Gender' in X_copy.columns:
            X_copy['Gender'] = X_copy['Gender'].astype('category').cat.codes
        
        # 3. Заполнение пропусков (те самые 243 строки)
        X_copy = X_copy.fillna(self.medians)
        
        return X_copy