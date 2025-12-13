from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import pandas as pd
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

if __name__ == "__main__":
    import uvicorn
    logger.info("=== ЗАПУСК FASTAPI СЕРВЕРА ===")
    logger.info("Сервер будет доступен по адресам:")
    logger.info("- http://localhost:8000/docs (проверка состояния)")
    logger.info("Для остановки сервера нажмите CTRL+C")
    logger.info("===============================")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)