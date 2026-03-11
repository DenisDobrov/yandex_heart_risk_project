import sys
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from catboost import CatBoostClassifier

# Добавляем пути
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.heart_risk.features.transformer import HeartFeatureTransformer

app = FastAPI(title="Heart Risk Prediction API")

# Глобальные объекты
model = CatBoostClassifier()
transformer = HeartFeatureTransformer()

@app.on_event("startup")
def load_assets():
    # Загружаем модель
    model_path = os.path.join(BASE_DIR, "artifacts", "model.cbm")
    if os.path.exists(model_path):
        model.load_model(model_path)
    
    # Чтобы трансформер знал медианы, "обучим" его на кусочке данных
    train_path = os.path.join(BASE_DIR, "data", "raw", "heart_train.csv") # проверьте имя файла!
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        transformer.fit(train_df)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 1. Читаем файл
        df = pd.read_csv(file.file)
        
        # Сохраняем ID для ответа
        ids = df['id'].tolist() if 'id' in df.columns else list(range(len(df)))
        
        # 2. Трансформируем
        # Удаляем таргет из теста, если он там вдруг есть
        if 'Heart Attack Risk (Binary)' in df.columns:
            df = df.drop(columns=['Heart Attack Risk (Binary)'])
            
        X = transformer.transform(df)
        
        # 3. Предсказание
        preds = model.predict(X)
        
        # 4. Ответ
        results = [
            {"id": int(id_val), "prediction": int(pred)} 
            for id_val, pred in zip(ids, preds)
        ]
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))