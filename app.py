from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
import numpy as np
import joblib
import os
import logging
from typing import List, Dict
from catboost import CatBoostClassifier
from io import StringIO
import traceback
from pydantic import BaseModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API для прогнозирования риска сердечного приступа на основе медицинских данных",
    version="1.0.0"
)

# Определение рабочей директории
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Текущая рабочая директория: {current_dir}")

# Глобальные переменные для модели и предпроцессора
model = None
preprocessor = None

class PredictionResult(BaseModel):
    id: int
    prediction: int

class APIResponse(BaseModel):
    predictions: List[PredictionResult]
    status: str
    details: str = ""

def load_artifacts():
    """Загрузка модели и предпроцессора с обработкой ошибок"""
    global model, preprocessor
    
    try:
        # Поиск файлов в разных директориях
        possible_model_paths = [
            "catboost_model.cbm",
            os.path.join(current_dir, "catboost_model.cbm"),
            os.path.join(current_dir, "stage_3", "catboost_model.cbm"),
            os.path.join(current_dir, "models", "catboost_model.cbm")
        ]
        
        possible_preproc_paths = [
            "preprocessor.pkl",
            os.path.join(current_dir, "preprocessor.pkl"),
            os.path.join(current_dir, "stage_2", "preprocessor.pkl"),
            os.path.join(current_dir, "models", "preprocessor.pkl")
        ]
        
        # Поиск и загрузка модели
        model_path = None
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            logger.info(f"Загрузка модели из: {model_path}")
            model = CatBoostClassifier()
            model.load_model(model_path)
            logger.info("✅ Модель успешно загружена")
        else:
            logger.error("❌ Файл модели не найден в указанных директориях")
            return
        
        # Поиск и загрузка предпроцессора
        preproc_path = None
        for path in possible_preproc_paths:
            if os.path.exists(path):
                preproc_path = path
                break
        
        if preproc_path:
            logger.info(f"Загрузка предпроцессора из: {preproc_path}")
            preprocessor = joblib.load(preproc_path)
            logger.info("✅ Предпроцессор успешно загружен")
        else:
            logger.error("❌ Файл предпроцессора не найден в указанных директориях")
            return
            
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке артефактов: {str(e)}")
        logger.error(traceback.format_exc())
        model = None
        preprocessor = None

# Загрузка артефактов при старте приложения
load_artifacts()

@app.post("/predict", response_model=APIResponse)
async def predict(file: UploadFile = File(...)):
    """
    Эндпоинт для прогнозирования риска сердечного приступа.
    Принимает CSV файл с медицинскими данными пациентов.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Сервис недоступен: модель или предпроцессор не загружены"
        )
    
    try:
        # Проверка типа файла
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400, 
                detail="Неверный формат файла. Требуется CSV файл."
            )
        
        # Чтение файла
        contents = await file.read()
        logger.info(f"Получен файл: {file.filename}, размер: {len(contents)} байт")
        
        # Чтение CSV без заголовков
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        logger.info(f"Загружено {len(df)} записей, форма данных: {df.shape}")
        
        # Проверка структуры данных
        if df.shape[1] < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Некорректная структура данных. Ожидается минимум 2 столбца (id + признаки), получено {df.shape[1]}"
            )
        
        # Обработка данных
        ids = df.iloc[:, 0].values
        features_df = df.iloc[:, 1:]
        
        logger.info(f"Идентификаторы: {len(ids)} записей")
        logger.info(f"Признаки для обработки: {features_df.shape}")
        
        # Предобработка данных
        logger.info("Предобработка данных...")
        X_processed = preprocessor.transform(features_df)
        logger.info(f"Форма обработанных данных: {X_processed.shape}")
        
        # Получение предсказаний
        logger.info("Генерация предсказаний...")
        predictions_proba = model.predict_proba(X_processed)[:, 1]
        predictions_class = (predictions_proba > 0.5).astype(int)
        
        # Формирование результата
        results = []
        for id_val, pred_class in zip(ids, predictions_class):
            results.append(PredictionResult(
                id=int(id_val),
                prediction=int(pred_class)
            ))
        
        logger.info(f"Сгенерировано {len(results)} предсказаний")
        
        # Создание CSV для скачивания
        results_df = pd.DataFrame([{"id": r.id, "prediction": r.prediction} for r in results])
        results_csv = results_df.to_csv(index=False)
        
        # Возвращаем JSON с предсказаниями и CSV для скачивания
        return APIResponse(
            predictions=results,
            status="success",
            details="Предсказания успешно сгенерированы"
        )
    
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке запроса: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Домашняя страница с информацией о сервисе"""
    status_html = f"""
    <div style="color: {'green' if model is not None and preprocessor is not None else 'red'}; font-weight: bold;">
        {'✅ СЕРВИС ГОТОВ К РАБОТЕ' if model is not None and preprocessor is not None else '❌ СЕРВИС НЕДОСТУПЕН'}
    </div>
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Heart Attack Risk Prediction API</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                line-height: 1.6;
            }}
            h1 {{
                color: #333;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
            }}
            .status {{
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
                background-color: #f8f9fa;
            }}
            .endpoints {{
                margin: 30px 0;
            }}
            .endpoint {{
                margin: 10px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            code {{
                background-color: #f1f1f1;
                padding: 2px 5px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>API для прогнозирования риска сердечного приступа</h1>
            
            <div class="status">
                <h2>Статус сервиса</h2>
                {status_html}
                <p><strong>Модель загружена:</strong> {'✅ Да' if model is not None else '❌ Нет'}</p>
                <p><strong>Предпроцессор загружен:</strong> {'✅ Да' if preprocessor is not None else '❌ Нет'}</p>
            </div>
            
            <div class="endpoints">
                <h2>Доступные эндпоинты</h2>
                
                <div class="endpoint">
                    <h3>POST /predict</h3>
                    <p>Прогнозирование риска сердечного приступа на основе CSV файла</p>
                    <p><strong>Параметры:</strong></p>
                    <ul>
                        <li><code>file</code>: CSV файл с данными пациентов (без заголовков, первый столбец - ID)</li>
                    </ul>
                    <p><strong>Формат ответа:</strong></p>
                    <pre>
{{
  "predictions": [
    {{"id": 1, "prediction": 0}},
    {{"id": 2, "prediction": 1}},
    ...
  ],
  "status": "success"
}}
                    </pre>
                </div>
                
                <div class="endpoint">
                    <h3>GET /docs</h3>
                    <p>Документация API (автоматически сгенерированная)</p>
                </div>
            </div>
            
            <div class="instructions">
                <h2>Инструкция по использованию</h2>
                <p>1. Перейдите на страницу <a href="/docs">/docs</a> для просмотра документации API</p>
                <p>2. Используйте эндпоинт <code>/predict</code> для отправки CSV файла с данными пациентов</p>
                <p>3. Скачайте CSV файл с предсказаниями из ответа</p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Эндпоинт для проверки здоровья сервиса"""
    return {
        "status": "healthy" if (model is not None and preprocessor is not None) else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "service_name": "Heart Attack Risk Prediction API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("=== ЗАПУСК FASTAPI СЕРВЕРА ===")
    logger.info("Сервер будет доступен по адресам:")
    logger.info("- http://localhost:8000 (главная страница)")
    logger.info("- http://localhost:8000/docs (документация API)")
    logger.info("- http://localhost:8000/health (проверка состояния)")
    logger.info("Для остановки сервера нажмите CTRL+C")
    logger.info("===============================")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)