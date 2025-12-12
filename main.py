from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from catboost import CatBoostClassifier
from io import StringIO, BytesIO
import traceback
import threading
import time
import requests
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI приложения
app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="API для прогнозирования риска сердечного приступа на основе медицинских данных",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Определение рабочей директории
current_dir = os.path.dirname(os.path.abspath(__file__))
logger.info(f"Текущая рабочая директория: {current_dir}")

# Глобальные переменные для модели и предпроцессора
model = None
preprocessor = None

# Pydantic модели для валидации
class PredictionResult(BaseModel):
    id: int
    prediction: int

class APIResponse(BaseModel):
    predictions: List[PredictionResult]
    status: str
    details: Optional[str] = ""

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    service_name: str
    version: str

# Классы в стиле ООП
class DataPreprocessor:
    """
    Класс для предобработки данных
    """
    def __init__(self):
        self.preprocessor = None
    
    def load(self, path: str):
        """
        Загрузка предпроцессора из файла
        """
        try:
            self.preprocessor = joblib.load(path)
            logger.info(f"Предпроцессор успешно загружен из {path}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при загрузке предпроцессора: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Применение преобразований к данным
        """
        try:
            if self.preprocessor is None:
                raise ValueError("Предпроцессор не загружен")
            
            # Кодирование категориального признака gender, если он есть
            gender_col = None
            for col in data.columns:
                if data[col].astype(str).str.contains('Male|Female').any():
                    gender_col = col
                    break
            
            if gender_col:
                # Кодирование пола: Male -> 0, Female -> 1
                gender_map = {'Male': 0, 'Female': 1}
                data[gender_col] = data[gender_col].map(gender_map).fillna(
                    data[gender_col].mode()[0] if not data[gender_col].mode().empty else 0
                )
            
            # Применение предпроцессора
            processed_data = self.preprocessor.transform(data)
            return processed_data
        except Exception as e:
            logger.error(f"Ошибка при обработке данных: {str(e)}")
            logger.error(traceback.format_exc())
            raise

class HeartAttackPredictor:
    """
    Класс для предсказания риска сердечного приступа
    """
    def __init__(self, model_path: str, preprocessor_path: str):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.feature_names = []
        
        # Загрузка модели и предпроцессора
        self._load_model()
        self._load_preprocessor()
    
    def _load_model(self):
        """
        Загрузка модели CatBoost
        """
        try:
            if not os.path.exists(self.model_path):
                # Поиск файла в разных директориях
                possible_paths = [
                    os.path.join(current_dir, self.model_path),
                    os.path.join(current_dir, "stage_3", self.model_path),
                    os.path.join(current_dir, "models", self.model_path)
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.model_path = path
                        break
            
            logger.info(f"Загрузка модели из: {self.model_path}")
            self.model = CatBoostClassifier()
            self.model.load_model(self.model_path)
            logger.info("✅ Модель успешно загружена")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {str(e)}")
            logger.error(traceback.format_exc())
            self.model = None
    
    def _load_preprocessor(self):
        """
        Загрузка предпроцессора
        """
        try:
            if not os.path.exists(self.preprocessor_path):
                # Поиск файла в разных директориях
                possible_paths = [
                    os.path.join(current_dir, self.preprocessor_path),
                    os.path.join(current_dir, "stage_2", self.preprocessor_path),
                    os.path.join(current_dir, "models", self.preprocessor_path)
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        self.preprocessor_path = path
                        break
            
            logger.info(f"Загрузка предпроцессора из: {self.preprocessor_path}")
            if not self.preprocessor.load(self.preprocessor_path):
                raise Exception("Не удалось загрузить предпроцессор")
            logger.info("✅ Предпроцессор успешно загружен")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке предпроцессора: {str(e)}")
            logger.error(traceback.format_exc())
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Генерация предсказаний
        """
        try:
            if self.model is None:
                raise ValueError("Модель не загружена")
            
            # Получение вероятностей
            probabilities = self.model.predict_proba(features)[:, 1]
            # Преобразование вероятностей в классы (порог 0.5)
            predictions = (probabilities > 0.5).astype(int)
            
            return predictions, probabilities
        except Exception as e:
            logger.error(f"❌ Ошибка при генерации предсказаний: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        """
        try:
            if self.model is None:
                raise ValueError("Модель не загружена")
            
            feature_importances = self.model.get_feature_importance()
            feature_names = [f"feature_{i}" for i in range(len(feature_importances))]
            
            # Сортировка важности признаков
            sorted_features = sorted(zip(feature_names, feature_importances), 
                                   key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "model_type": "CatBoostClassifier",
                "best_iteration": self.model.get_best_iteration() if hasattr(self.model, 'get_best_iteration') else None,
                "eval_metric": self.model.get_params().get('eval_metric', 'AUC'),
                "feature_count": len(feature_importances),
                "top_features": [
                    {"feature": name, "importance": float(imp)} 
                    for name, imp in sorted_features
                ]
            }
        except Exception as e:
            logger.error(f"❌ Ошибка при получении информации о модели: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Инициализация предиктора
predictor = None

def init_predictor():
    """
    Инициализация предиктора при запуске приложения
    """
    global predictor
    
    # Поиск файлов модели и предпроцессора
    model_path = "catboost_model.cbm"
    preprocessor_path = "preprocessor.pkl"
    
    # Инициализация предиктора
    predictor = HeartAttackPredictor(model_path, preprocessor_path)

@app.on_event("startup")
async def startup_event():
    """
    Инициализация при запуске приложения
    """
    logger.info("=== ИНИЦИАЛИЗАЦИЯ ПРИЛОЖЕНИЯ ===")
    init_predictor()
    logger.info("=== ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА ===")

@app.post("/predict", response_model=APIResponse)
async def predict(file: UploadFile = File(...), return_csv: bool = False):
    """
    Эндпоинт для прогнозирования риска сердечного приступа.
    Принимает CSV файл с медицинскими данными пациентов.
    
    Параметры:
    - file: CSV файл с данными
    - return_csv: если True, вернет CSV файл вместо JSON
    
    Возвращает:
    - JSON с предсказаниями или CSV файл с колонками "id" и "prediction"
    """
    if predictor is None or predictor.model is None or predictor.preprocessor.preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Сервис временно недоступен. Модель или предпроцессор не загружены."
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
        
        # Автоматическое определение кодировки
        from chardet import detect
        result = detect(contents)
        encoding = result['encoding'] or 'utf-8'
        logger.info(f"Определена кодировка файла: {encoding}")
        
        # Чтение CSV без заголовков
        df = pd.read_csv(StringIO(contents.decode(encoding)))
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
        X_processed = predictor.preprocessor.transform(features_df)
        logger.info(f"Форма обработанных данных: {X_processed.shape}")
        
        # Получение предсказаний
        logger.info("Генерация предсказаний...")
        predictions_class, predictions_proba = predictor.predict(X_processed)
        
        # Формирование результата
        results = []
        for id_val, pred_class in zip(ids, predictions_class):
            results.append(PredictionResult(
                id=int(id_val),
                prediction=int(pred_class)
            ))
        
        logger.info(f"Сгенерировано {len(results)} предсказаний")
        
        # Если нужно вернуть CSV файл
        if return_csv:
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, "predictions.csv")
            
            # Создаем DataFrame с результатами
            result_df = pd.DataFrame([{
                "id": r.id, 
                "prediction": r.prediction
            } for r in results])
            
            # Сохраняем в CSV
            result_df.to_csv(output_path, index=False)
            logger.info(f"Результаты сохранены в {output_path}")
            
            return FileResponse(
                output_path, 
                media_type='text/csv', 
                filename="predictions.csv",
                headers={"Content-Disposition": "attachment; filename=predictions.csv"}
            )
        
        # Возвращаем JSON
        return APIResponse(
            predictions=results,
            status="success",
            details=f"Успешно обработано {len(results)} записей"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ошибка при обработке запроса: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Эндпоинт для проверки состояния сервиса
    """
    return HealthResponse(
        status="healthy" if (predictor is not None and predictor.model is not None and predictor.preprocessor.preprocessor is not None) else "unhealthy",
        model_loaded=predictor is not None and predictor.model is not None,
        preprocessor_loaded=predictor is not None and predictor.preprocessor.preprocessor is not None,
        service_name="Heart Attack Risk Prediction API",
        version="1.0.0"
    )

@app.get("/model-info")
async def model_info():
    """
    Эндпоинт для получения информации о модели
    """
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
    try:
        info = predictor.get_model_info()
        return info
    except Exception as e:
        logger.error(f"Ошибка при получении информации о модели: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Ошибка при получении информации о модели: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Домашняя страница с информацией о сервисе
    """
    status_html = f"""
    <div style="color: {'green' if (predictor is not None and predictor.model is not None and predictor.preprocessor.preprocessor is not None) else 'red'}; font-weight: bold; font-size: 24px;">
        {'✅ СЕРВИС ГОТОВ К РАБОТЕ' if (predictor is not None and predictor.model is not None and predictor.preprocessor.preprocessor is not None) else '❌ СЕРВИС НЕДОСТУПЕН'}
    </div>
    """
    
    model_status = "✅ Загружена" if (predictor is not None and predictor.model is not None) else "❌ Не загружена"
    preprocessor_status = "✅ Загружен" if (predictor is not None and predictor.preprocessor.preprocessor is not None) else "❌ Не загружен"
    
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
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
            }}
            .status {{
                padding: 20px;
                margin: 30px 0;
                border-radius: 8px;
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
            }}
            .endpoints {{
                margin: 40px 0;
            }}
            .endpoint {{
                margin: 20px 0;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
                transition: box-shadow 0.3s ease;
            }}
            .endpoint:hover {{
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .endpoint h3 {{
                color: #3498db;
                margin-top: 0;
            }}
            code {{
                background-color: #f1f1f1;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: monospace;
                font-size: 14px;
            }}
            pre {{
                background-color: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-family: monospace;
                margin: 15px 0;
            }}
            .example {{
                background-color: #e8f4f8;
                padding: 15px;
                border-radius: 5px;
                margin: 15px 0;
            }}
            .links {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-top: 30px;
            }}
            .link-item {{
                flex: 1;
                min-width: 200px;
            }}
            a.button {{
                display: inline-block;
                padding: 10px 20px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                font-weight: bold;
                transition: background-color 0.3s ease;
            }}
            a.button:hover {{
                background-color: #2980b9;
            }}
            @media (max-width: 768px) {{
                .links {{
                    flex-direction: column;
                }}
                a.button {{
                    width: 100%;
                    text-align: center;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>API для прогнозирования риска сердечного приступа</h1>
            
            <div class="status">
                <h2>📊 Текущий статус сервиса</h2>
                {status_html}
                <p><strong>Модель:</strong> {model_status}</p>
                <p><strong>Предпроцессор:</strong> {preprocessor_status}</p>
                <p><strong>Версия API:</strong> 1.0.0</p>
            </div>
            
            <div class="endpoints">
                <h2>🔧 Доступные эндпоинты</h2>
                
                <div class="endpoint">
                    <h3>POST /predict</h3>
                    <p>Прогнозирование риска сердечного приступа на основе CSV файла</p>
                    <p><strong>Параметры запроса:</strong></p>
                    <ul>
                        <li><code>file</code>: CSV файл с данными пациентов (обязательный)</li>
                        <li><code>return_csv</code>: true/false - вернуть результат в формате CSV (опционально, по умолчанию false)</li>
                    </ul>
                    <p><strong>Формат данных в CSV:</strong></p>
                    <ul>
                        <li>Файл без заголовков</li>
                        <li>Первый столбец: ID пациента</li>
                        <li>Остальные столбцы: медицинские признаки</li>
                    </ul>
                    <p><strong>Пример ответа (JSON):</strong></p>
                    <pre>
{{
  "predictions": [
    {{"id": 1, "prediction": 0}},
    {{"id": 2, "prediction": 1}},
    ...
  ],
  "status": "success",
  "details": "Успешно обработано 10 записей"
}}
                    </pre>
                </div>
                
                <div class="endpoint">
                    <h3>GET /health</h3>
                    <p>Проверка состояния сервиса</p>
                    <p><strong>Пример ответа:</strong></p>
                    <pre>
{{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true,
  "service_name": "Heart Attack Risk Prediction API",
  "version": "1.0.0"
}}
                    </pre>
                </div>
                
                <div class="endpoint">
                    <h3>GET /model-info</h3>
                    <p>Получение информации о модели</p>
                    <p><strong>Пример ответа:</strong></p>
                    <pre>
{{
  "model_type": "CatBoostClassifier",
  "best_iteration": 125,
  "eval_metric": "AUC",
  "feature_count": 26,
  "top_features": [
    {{"feature": "feature_23", "importance": 0.15}},
    {{"feature": "feature_5", "importance": 0.12}},
    ...
  ]
}}
                    </pre>
                </div>
            </div>
            
            <div class="instructions">
                <h2>🚀 Инструкция по использованию</h2>
                
                <h3>1. Через Swagger UI (рекомендуется)</h3>
                <p>Перейдите на страницу <a href="/docs" class="button">Документация API</a> и используйте интерактивный интерфейс для тестирования эндпоинтов.</p>
                
                <h3>2. Через curl</h3>
                <p><strong>Отправка файла и получение JSON:</strong></p>
                <pre>curl -X POST "http://localhost:8000/predict" -H "Content-Type: multipart/form-data" -F "file=@heart_test.csv"</pre>
                
                <p><strong>Отправка файла и получение CSV:</strong></p>
                <pre>curl -X POST "http://localhost:8000/predict?return_csv=true" -H "Content-Type: multipart/form-data" -F "file=@heart_test.csv" --output predictions.csv</pre>
                
                <h3>3. Через Python</h3>
                <pre>import requests

url = "http://localhost:8000/predict"
with open('heart_test.csv', 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)
    
print(response.json())</pre>
            </div>
            
            <div class="links">
                <div class="link-item">
                    <a href="/docs" class="button">📚 Документация API (Swagger UI)</a>
                </div>
                <div class="link-item">
                    <a href="/redoc" class="button">📖 Альтернативная документация (ReDoc)</a>
                </div>
                <div class="link-item">
                    <a href="/health" class="button">✅ Проверка состояния</a>
                </div>
                <div class="link-item">
                    <a href="/model-info" class="button">📊 Информация о модели</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

def run_server_in_thread():
    """
    Запуск сервера в отдельном потоке для совместимости с Jupyter
    """
    import uvicorn
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    import uvicorn
    
    # Информационное сообщение
    logger.info("=== ЗАПУСК FASTAPI СЕРВЕРА ===")
    logger.info("Сервер будет доступен по адресам:")
    logger.info("- http://localhost:8000/docs (документация API)")
    logger.info("- http://localhost:8000/health (проверка состояния)")
    logger.info("Для остановки сервера нажмите CTRL+C")
    logger.info("===============================")
    
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)