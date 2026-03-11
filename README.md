# yandex_heart_risk_project

Сервис на базе **FastAPI** и **CatBoost** для классификации пациентов по уровню риска.

## Как запустить проект
1. Установите зависимости: `uv pip install -r requirements.txt`
2. Запустите сервер: `uvicorn app.main:app`
3. Откройте документацию: `http://127.0.0.1:8000/docs`


### Структура проекта
* `src/heart_risk/` — основная библиотека (Data Loaders, Transformers, Trainers)
* `app/` — веб-сервис на FastAPI
* `artifacts/` — сохраненные веса модели (`model.cbm`)
* `notebooks/` — исследование данных (EDA) и эксперименты
* `predictions/` — финальные предсказания на тестовой выборке