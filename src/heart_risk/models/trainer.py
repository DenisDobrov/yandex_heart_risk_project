from catboost import CatBoostClassifier
import pandas as pd

class HeartModelTrainer:
    def __init__(self, iterations=1000, learning_rate=0.03):
        # Рассчитаем примерный вес: здоровых 0.65 / больных 0.35 = ~1.85
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=6,
            l2_leaf_reg=3,
            # Прямой коэффициент для усиления внимания к классу 1.0
            scale_pos_weight=1.85, 
            eval_metric='F1',
            early_stopping_rounds=50,
            verbose=100,
            random_seed=42
        )

    def train(self, X, y):
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val)
        )
        return self.model

    def evaluate(self, X, y):
        from sklearn.metrics import classification_report
        preds = self.model.predict(X)
        print(classification_report(y, preds))