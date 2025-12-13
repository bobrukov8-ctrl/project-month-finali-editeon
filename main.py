from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from catboost import CatBoostClassifier
from io import StringIO
import traceback

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Инициализация приложения
app = FastAPI(
    title="Heart Attack Risk Prediction API",
    description="Сервис прогнозирования сердечно-сосудистых заболеваний",
    version="1.1.0"
)

# --- CONFIG & GLOBALS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = "catboost_model.cbm"
PREPROCESSOR_FILE = "preprocessor.pkl"

# Глобальный объект предиктора
predictor = None

# --- PYDANTIC MODELS ---
class PredictionResult(BaseModel):
    id: int
    prediction: int

class APIResponse(BaseModel):
    predictions: List[PredictionResult]
    status: str
    details: Optional[str] = ""

# --- CORE CLASSES ---
class DataPreprocessor:
    def __init__(self):
        self.pipeline = None
    
    def load(self, path: str) -> bool:
        try:
            self.pipeline = joblib.load(path)
            logger.info(f"✅ Предпроцессор загружен: {path}")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки предпроцессора: {e}")
            return False
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if not self.pipeline:
            raise ValueError("Предпроцессор не инициализирован")
        
        # --- ИСПРАВЛЕНИЕ ОШИБКИ ---
        # Если у модели сохранены имена признаков, применяем их к входящим данным
        if hasattr(self.pipeline, 'feature_names_in_'):
            expected_features = self.pipeline.feature_names_in_
            
            # Проверяем, совпадает ли количество колонок
            if len(data.columns) == len(expected_features):
                # Создаем копию, чтобы не менять исходный объект и избежать предупреждений
                data = data.copy()
                data.columns = expected_features
                # logger.info("Названия колонок восстановлены из предпроцессора")
            else:
                logger.warning(f"Внимание: число колонок ({len(data.columns)}) не совпадает с ожидаемым ({len(expected_features)})!")
        # ---------------------------

        # Обработка пола (Male/Female -> 0/1) если есть такие колонки
        # Теперь поиск будет работать корректно, так как мы восстановили имена
        for col in data.columns:
            if data[col].dtype == 'object' and data[col].astype(str).str.contains('Male|Female').any():
                data[col] = data[col].map({'Male': 0, 'Female': 1}).fillna(0)
        
        return self.pipeline.transform(data)

class HeartAttackPredictor:
    def __init__(self):
        self.model = CatBoostClassifier()
        self.preprocessor = DataPreprocessor()
        self.is_ready = False
        self.model_loaded = False
        self.preprocessor_loaded = False

    def initialize(self):
        """Пытается найти и загрузить модель и предпроцессор"""
        # Поиск файлов (текущая папка, models, stage_3 и т.д.)
        search_dirs = [current_dir, os.path.join(current_dir, "models"), os.path.join(current_dir, "stage_3")]
        
        # Загрузка модели
        for path in search_dirs:
            m_path = os.path.join(path, MODEL_FILE)
            if os.path.exists(m_path):
                try:
                    self.model.load_model(m_path)
                    self.model_loaded = True
                    logger.info(f"✅ Модель загружена: {m_path}")
                    break
                except Exception as e:
                    logger.error(f"Ошибка модели: {e}")

        # Загрузка предпроцессора
        for path in search_dirs:
            p_path = os.path.join(path, PREPROCESSOR_FILE)
            if os.path.exists(p_path):
                if self.preprocessor.load(p_path):
                    self.preprocessor_loaded = True
                    break
        
        self.is_ready = self.model_loaded and self.preprocessor_loaded

    def predict_batch(self, df: pd.DataFrame) -> List[int]:
        if not self.is_ready:
            raise RuntimeError("Система не готова (модель или предпроцессор не загружены)")
        
        X_processed = self.preprocessor.transform(df)
        # Получаем вероятности, берем класс 1 если вероятность > 0.5
        probs = self.model.predict_proba(X_processed)[:, 1]
        return (probs > 0.5).astype(int).tolist()

# --- EVENTS ---
@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = HeartAttackPredictor()
    predictor.initialize()

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Оптимизированная главная страница с панелью управления
    """
    # Состояние системы
    is_model_ok = predictor.model_loaded
    is_prep_ok = predictor.preprocessor_loaded
    
    status_color = "#2ecc71" if (is_model_ok and is_prep_ok) else "#e74c3c"
    status_text = "СИСТЕМА ГОТОВА" if (is_model_ok and is_prep_ok) else "ТРЕБУЕТСЯ ВНИМАНИЕ"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Heart Risk AI Dashboard</title>
        <style>
            :root {{ --primary: #3498db; --success: #2ecc71; --danger: #e74c3c; --bg: #f4f6f9; }}
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: var(--bg); margin: 0; padding: 20px; color: #333; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
            
            h1 {{ text-align: center; color: #2c3e50; margin-bottom: 30px; }}
            
            /* Status Cards */
            .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }}
            .card {{ padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #eee; }}
            .card.main-status {{ background: {status_color}; color: white; font-weight: bold; grid-column: 1 / -1; }}
            .indicator {{ font-size: 24px; margin-bottom: 10px; display: block; }}
            .badge-ok {{ color: var(--success); background: #eafaf1; padding: 5px 10px; border-radius: 15px; }}
            .badge-err {{ color: var(--danger); background: #fdeaea; padding: 5px 10px; border-radius: 15px; }}

            /* Upload Section */
            .upload-zone {{ border: 2px dashed #bdc3c7; padding: 40px; text-align: center; border-radius: 8px; transition: 0.3s; background: #fafafa; }}
            .upload-zone:hover {{ border-color: var(--primary); background: #ecf5fb; }}
            input[type="file"] {{ display: none; }}
            .btn {{ background: var(--primary); color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 10px; transition: 0.2s; }}
            .btn:hover {{ background: #2980b9; }}
            .btn:disabled {{ background: #95a5a6; cursor: not-allowed; }}
            
            /* Instructions */
            .steps {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-top: 40px; }}
            .step-item {{ display: flex; align-items: center; margin-bottom: 10px; }}
            .step-num {{ background: var(--primary); color: white; width: 25px; height: 25px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-weight: bold; }}
            
            #resultArea {{ margin-top: 20px; padding: 15px; border-radius: 8px; display: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Анализ Риска Сердечного Приступа</h1>
            
            <div class="status-grid">
                <div class="card main-status">
                    {status_text}
                </div>
                <div class="card">
                    <span class="indicator">🧠</span>
                    <div>Модель: <span class="{'badge-ok' if is_model_ok else 'badge-err'}">{'ЗАГРУЖЕНА' if is_model_ok else 'ОШИБКА'}</span></div>
                </div>
                <div class="card">
                    <span class="indicator">⚙️</span>
                    <div>Предпроцессор: <span class="{'badge-ok' if is_prep_ok else 'badge-err'}">{'ЗАГРУЖЕН' if is_prep_ok else 'ОШИБКА'}</span></div>
                </div>
            </div>

            <div class="upload-zone">
                <h2>Быстрая проверка</h2>
                <p>Загрузите CSV файл для получения прогноза прямо сейчас</p>
                <form id="apiForm">
                    <label for="csvFile" class="btn">📂 Выбрать файл</label>
                    <input type="file" id="csvFile" name="file" accept=".csv" onchange="updateFileName()">
                    <span id="fileName" style="margin-left: 10px; color: #7f8c8d;">Файл не выбран</span>
                    <br><br>
                    <input type="checkbox" id="returnCsv" name="return_csv" style="margin-right: 5px;">
                    <label for="returnCsv" style="font-weight: bold;">Скачать результат как CSV-файл</label>
                    <br><br>
                    <button type="button" class="btn" onclick="sendPrediction()" {'disabled' if not (is_model_ok and is_prep_ok) else ''}>🚀 Получить прогноз</button>
                </form>
                <div id="resultArea"></div>
            </div>

            <div class="steps">
                <h3>📋 Как это работает:</h3>
                <div class="step-item">
                    <div class="step-num">1</div>
                    <div>Подготовьте <strong>CSV файл</strong> без заголовков (ID, Feature1, Feature2...)</div>
                </div>
                <div class="step-item">
                    <div class="step-num">2</div>
                    <div>Нажмите <strong>"Выбрать файл"</strong> выше или используйте API <code>/predict</code></div>
                </div>
                <div class="step-item">
                    <div class="step-num">3</div>
                    <div>Получите JSON с результатами или скачайте CSV отчет.</div>
                </div>
                <p style="margin-top: 15px; font-size: 0.9em;">
                    <a href="/docs">Техническая документация (Swagger)</a> | 
                    <a href="/redoc">ReDoc</a>
                </p>
            </div>
        </div>

        <script>
            function updateFileName() {{
                const input = document.getElementById('csvFile');
                const span = document.getElementById('fileName');
                if(input.files.length > 0) {{
                    span.textContent = input.files[0].name;
                }}
            }}

            async function sendPrediction() {{
                const input = document.getElementById('csvFile');
                const returnCsv = document.getElementById('returnCsv').checked;
                const resultArea = document.getElementById('resultArea');
                
                if(input.files.length === 0) {{
                    alert("Пожалуйста, выберите файл!");
                    return;
                }}

                const formData = new FormData();
                formData.append("file", input.files[0]);

                resultArea.style.display = 'block';
                resultArea.innerHTML = "⏳ Обработка данных...";
                resultArea.style.background = '#fff3cd';

                const url = `/predict?return_csv=${{returnCsv}}`;

                try {{
                    const response = await fetch(url, {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    if(response.ok) {{
                        if (returnCsv) {{
                            // Режим скачивания CSV
                            const blob = await response.blob();
                            const downloadUrl = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = downloadUrl;
                            a.download = 'predictions.csv';
                            document.body.appendChild(a);
                            a.click();
                            a.remove();
                            window.URL.revokeObjectURL(downloadUrl);
                            
                            resultArea.style.background = '#d4edda';
                            resultArea.innerHTML = `<strong>✅ Готово!</strong><br>Файл <strong>predictions.csv</strong> загружен.`;
                            
                        }} else {{
                            // Режим отображения JSON
                            const data = await response.json();
                            resultArea.style.background = '#d4edda';
                            resultArea.innerHTML = `<strong>✅ Готово!</strong><br>Обработано записей: ${{data.predictions.length}}<br><br><pre style="text-align:left; max-height:200px; overflow:auto;">${{JSON.stringify(data.predictions, null, 2)}}</pre>`;
                        }}
                    }} else {{
                        // Обработка ошибок в обоих режимах
                        const errorText = await response.text();
                        resultArea.style.background = '#f8d7da';
                        resultArea.innerHTML = `❌ Ошибка (${{response.status}}): ${{errorText}}`;
                    }}
                }} catch (error) {{
                    resultArea.style.background = '#f8d7da';
                    resultArea.innerHTML = `❌ Ошибка сети: ${{error.message}}`;
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=APIResponse)
async def predict(file: UploadFile = File(...), return_csv: bool = False):
    """
    Основной эндпоинт прогнозирования
    """
    if not predictor.is_ready:
        raise HTTPException(503, "Модель не загружена. Проверьте статус на главной странице.")

    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Ожидается файл .csv")

    try:
        content = await file.read()
        
        # Декодирование файла
        try:
            s = str(content, 'utf-8')
        except UnicodeDecodeError:
            s = str(content, 'latin-1')

        # Чтение данных (автоматическое определение заголовка)
        df = pd.read_csv(StringIO(s), header=None)
        
        first_cell = str(df.iloc[0, 0]).lower()
        if 'id' in first_cell or 'age' in first_cell or df.shape[1] == 28: # Добавил проверку на 28 колонок
            df = pd.read_csv(StringIO(s))
            logger.info("Обнаружен заголовок в CSV файле.")
        
        # --- НОВОЕ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ---
        
        # Получаем ожидаемое количество признаков из обученного предпроцессора
        expected_feature_count = 0
        if predictor.preprocessor.pipeline and hasattr(predictor.preprocessor.pipeline, 'feature_names_in_'):
            expected_feature_count = len(predictor.preprocessor.pipeline.feature_names_in_)
        
        if expected_feature_count == 0:
            # Если не смогли определить ожидаемое число, возвращаем ошибку, т.к. это критично
            raise RuntimeError("Не удалось определить ожидаемое число признаков из preprocessor.pkl.")

        # Проверка, что после извлечения ID осталось минимум столько же колонок
        if df.shape[1] < expected_feature_count + 1:
             raise HTTPException(400, f"Ошибка: Файл должен содержать ID + {expected_feature_count} признаков. Обнаружено всего {df.shape[1]} колонок.")

        # Отделяем ID (1-й столбец, индекс 0)
        ids = df.iloc[:, 0].values
        
        # Выбираем только точное количество признаков (26): колонки с 1 по 26 включительно
        # Срез [1 : 1 + expected_feature_count]
        features = df.iloc[:, 1 : 1 + expected_feature_count]

        # Окончательная проверка среза
        if features.shape[1] != expected_feature_count:
             # Эта ошибка должна предотвратить 'NoneType' в sklearn
             raise HTTPException(400, f"Ошибка: Предпроцессор ожидает {expected_feature_count} признаков, но файл содержит {features.shape[1]} после извлечения ID. Проверьте, что в файле нет лишнего столбца (например, целевой переменной).")

        # --- КОНЕЦ КРИТИЧЕСКОГО ИСПРАВЛЕНИЯ ---

        # Предсказание
        predictions = predictor.predict_batch(features)
        
        # ... (остальной код остается прежним)
        results = [PredictionResult(id=int(i), prediction=p) for i, p in zip(ids, predictions)]

        if return_csv:
            output_df = pd.DataFrame([{"id": r.id, "prediction": r.prediction} for r in results])
            stream = StringIO()
            output_df.to_csv(stream, index=False)
            response = HTMLResponse(stream.getvalue(), media_type="text/csv")
            response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
            return response

        return APIResponse(
            predictions=results,
            status="success",
            details=f"Обработано {len(results)} строк"
        )

    except HTTPException:
        # Переброс HTTPException, чтобы избежать обработки ниже
        raise
    except Exception as e:
        logger.error(f"Ошибка в /predict: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(500, f"Ошибка обработки: {str(e)}")

print ("перейдите по ссылке http://localhost:8000/")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)