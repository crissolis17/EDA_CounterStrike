# Counter Strike ML Predictor

Este proyecto es una aplicación web interactiva que utiliza modelos de Machine Learning para predecir la supervivencia de jugadores en partidas de Counter Strike: Global Offensive (CS:GO). Incluye un backend en Python (Flask) que expone modelos de regresión y clasificación, y un frontend moderno para la interacción del usuario.

---

## Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos y Dependencias](#requisitos-y-dependencias)
- [Instalación y Ejecución](#instalación-y-ejecución)
- [Uso de la Aplicación](#uso-de-la-aplicación)
- [Endpoints de la API](#endpoints-de-la-api)
- [Características Usadas por los Modelos](#características-usadas-por-los-modelos)
- [Lógica de Predicción](#lógica-de-predicción)
- [Notas sobre el Dataset](#notas-sobre-el-dataset)
- [Créditos](#créditos)

---

## Descripción General

Esta aplicación permite predecir, a partir de variables clave de una ronda de CS:GO (mapa, equipamiento, arma principal), el tiempo estimado de supervivencia y la probabilidad de sobrevivir la ronda. Utiliza dos modelos:
- **Random Forest (Regresión):** Predice el tiempo de supervivencia en segundos.
- **XGBoost (Clasificación):** Predice si la supervivencia será alta o baja.

El frontend permite al usuario configurar los parámetros y visualizar los resultados y análisis de forma gráfica e intuitiva.

---

## Estructura del Proyecto

```
counter_strike_ml_web/
├── backend/           # Backend en Flask + modelos ML
│   ├── app.py         # API principal
│   ├── models/        # Modelos y features serializados
│   ├── ...            # Scripts de entrenamiento y utilidades
├── frontend/          # Interfaz web
│   ├── index.html     # Página principal
│   ├── js/            # Lógica JS
│   └── css/           # Estilos
```

---

## Requisitos y Dependencias

### Backend (Python 3.8+)
- Flask
- flask-cors
- pandas
- numpy
- scikit-learn
- xgboost

Instala las dependencias ejecutando:
```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost
```

### Frontend
Solo necesitas un navegador web moderno (Chrome, Firefox, Edge, etc.).

---

## Instalación y Ejecución

### 1. Ejecutar el Backend
Desde la carpeta `backend/`:
```bash
python app.py
```
Esto levantará la API en `http://localhost:5000`.

### 2. Abrir el Frontend
Abre el archivo `frontend/index.html` en tu navegador. Puedes hacerlo directamente o servirlo con un servidor estático (opcional).

---

## Uso de la Aplicación

1. Selecciona el mapa, el valor de equipamiento personal y de equipo, y el arma principal.
2. Haz clic en los botones para obtener:
   - **Predicción de Tiempo (Random Forest):** Tiempo estimado de supervivencia en segundos.
   - **Clasificación de Supervivencia (XGBoost):** Probabilidad de sobrevivir la ronda (alta/baja).
3. Visualiza los resultados, análisis y gráficos generados.

---

## Endpoints de la API

### 1. `/api/health` (GET)
Verifica el estado del backend y los modelos.

**Respuesta ejemplo:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "timestamp": "2025-05-01T12:34:56.789Z",
  "random_forest_performance": "R² = 13.16% (mejorado 130x)",
  "xgboost_performance": "97.31% accuracy",
  "dataset": "Anexo ET_demo_round_traces_2022 (1).csv",
  "timestamp_conversion": "Logarítmica (original → CS:GO time)"
}
```

### 2. `/api/predict/regression` (POST)
Predice el tiempo de supervivencia usando Random Forest.

**Body JSON:**
```json
{
  "map": "de_dust2",
  "equipment": 4000,
  "teamEquipment": 16000,
  "weapon": "rifle"
}
```
**Respuesta ejemplo:**
```json
{
  "prediction": 62.3,
  "confidence": 0.65,
  "model": "Random Forest (Regresión mejorada)",
  "interpretation": "Buena supervivencia - Situación favorable",
  "confidence_level": "Moderada (R² = 13.16%)",
  "raw_timestamp": 61.8
}
```

### 3. `/api/predict/classification` (POST)
Predice la probabilidad de supervivencia usando XGBoost.

**Body JSON:**
```json
{
  "map": "de_dust2",
  "equipment": 4000,
  "teamEquipment": 16000,
  "weapon": "rifle"
}
```
**Respuesta ejemplo:**
```json
{
  "prediction": 1,
  "probability_low": 0.22,
  "probability_high": 0.78,
  "confidence": 0.76,
  "label": "Alta Supervivencia",
  "model": "XGBoost",
  "interpretation": "Probabilidad de sobrevivir la ronda (Confianza: Alta)",
  "confidence_level": "Muy Alta (97.31% accuracy)"
}
```

### 4. `/api/test-prediction` (GET)
Devuelve una predicción de ejemplo para pruebas rápidas.

---

## Características Usadas por los Modelos

Ambos modelos utilizan variables clave del contexto de la ronda:
- **map:** Mapa de la ronda (`de_dust2`, `de_inferno`, `de_mirage`, `de_nuke`)
- **equipment:** Valor de equipamiento personal (0-16000)
- **teamEquipment:** Valor de equipamiento del equipo (0-80000)
- **weapon:** Arma principal (`rifle`, `sniper`, `smg`, `pistol`, `heavy`)

Internamente, los modelos usan un vector de características derivadas, incluyendo:
- Equipamiento personal y de equipo
- Tipo de arma (one-hot encoding)
- Mapa (codificado)
- Variables neutras para kills y headshots (para evitar sesgo)
- Derivadas: distancia recorrida, granadas lanzadas, ratios de eficiencia, etc.

**Lista completa de features:**
- MatchKills, RoundKills, MatchAssists, RoundAssists
- MatchHeadshots, RoundHeadshots, MatchFlankKills, RoundFlankKills
- RoundStartingEquipmentValue, TeamStartingEquipmentValue
- TravelledDistance, FirstKillTime, RLethalGrenadesThrown, RNonLethalGrenadesThrown
- PrimaryAssaultRifle, PrimarySniperRifle, PrimaryHeavy, PrimarySMG, PrimaryPistol
- KD_Ratio, Headshot_Efficiency, Equipment_ROI, Assist_Ratio
- Map_Encoded, Team_Encoded

---

## Lógica de Predicción

- **Random Forest (Regresión):**
  - Predice el tiempo de supervivencia en segundos.
  - La confianza se basa en el R² del modelo (13.16%).
  - Interpreta el resultado en categorías: muy rápida, rápida, moderada, buena o excelente supervivencia.

- **XGBoost (Clasificación):**
  - Predice si la supervivencia será alta (1) o baja (0).
  - Devuelve probabilidades para ambas clases y un nivel de confianza.
  - Interpreta el resultado y da recomendaciones según el riesgo.

---

## Notas sobre el Dataset

- El dataset real usado para entrenar los modelos es `Anexo ET_demo_round_traces_2022 (1).csv`.
- Para pruebas y desarrollo, se pueden generar datos sintéticos realistas con el script `create_realistic_models.py`.
- La lógica de los modelos prioriza el equipamiento, el tipo de arma y el contexto del mapa para evitar sesgos por kills o headshots.

---

## Créditos

Proyecto académico para demostración de técnicas de Machine Learning aplicadas a videojuegos (CS:GO).

Desarrollado por: Cristian y colaboradores.

---

© 2025 Counter Strike ML Predictor 