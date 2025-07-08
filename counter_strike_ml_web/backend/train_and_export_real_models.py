#!/usr/bin/env python3
# =============================================================================
# SCRIPT PARA ENTRENAR Y EXPORTAR MODELOS REALES CON TU DATASET
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, r2_score, mean_squared_error,
                             mean_absolute_error, cross_val_score, roc_curve)

warnings.filterwarnings('ignore')

# Instalar XGBoost si es necesario
try:
    import xgboost as xgb
    print("✅ XGBoost disponible")
except ImportError:
    print("📦 Instalando XGBoost...")
    import subprocess
    subprocess.run(["pip", "install", "xgboost"])
    import xgboost as xgb

print("🚀 ENTRENAMIENTO DE MODELOS REALES CON TU DATASET")
print("="*60)
print("📊 Usando: Anexo ET_demo_round_traces_2022 (1).csv")
print("🎯 Modelos: XGBoost (Clasificación) + Random Forest (Regresión)")
print("="*60)

# Verificar que el dataset existe
dataset_path = 'Anexo ET_demo_round_traces_2022 (1).csv'
if not os.path.exists(dataset_path):
    print(f"❌ No se encuentra el dataset: {dataset_path}")
    print("📁 Por favor, copia el archivo a la carpeta backend/")
    exit(1)

print(f"✅ Dataset encontrado: {dataset_path}")

# =============================================================================
# PARTE 1: EJECUTAR TU CÓDIGO DE XGBOOST (CLASIFICACIÓN)
# =============================================================================

print("\n" + "="*60)
print("PARTE 1: ENTRENANDO XGBOOST (CLASIFICACIÓN)")
print("="*60)

# Función para cargar y convertir datos (de tu código XGBoost)


def load_and_convert_csgo_data():
    """Cargar y convertir timestamps específicos de CS:GO"""
    print("\n📂 PASO 1: CARGA Y CONVERSIÓN DE TIMESTAMPS")
    print("-" * 42)

    # Cargar datos
    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
    print(
        f"✅ Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")

    # Función para convertir timestamps de CS:GO
    def convert_csgo_timestamp(timestamp_str):
        """Convierte timestamps específicos de CS:GO a segundos"""
        if pd.isna(timestamp_str):
            return np.nan

        try:
            # Remover puntos (separadores de miles)
            cleaned = str(timestamp_str).replace('.', '')
            timestamp_num = float(cleaned)

            # Determinar unidad basándose en magnitud
            if timestamp_num > 1e15:  # Nanosegundos
                seconds = timestamp_num / 1e9
            elif timestamp_num > 1e12:  # Microsegundos
                seconds = timestamp_num / 1e6
            elif timestamp_num > 1e9:   # Milisegundos
                seconds = timestamp_num / 1e3
            else:  # Ya en segundos
                seconds = timestamp_num

            return seconds

        except (ValueError, TypeError):
            return np.nan

    # Convertir TimeAlive
    print("🔄 Convirtiendo timestamps...")
    df['TimeAlive_seconds'] = df['TimeAlive'].apply(convert_csgo_timestamp)

    # Verificar conversión
    valid_conversions = df['TimeAlive_seconds'].notna().sum()
    conversion_rate = valid_conversions / len(df) * 100

    print(f"✅ Conversiones exitosas: {valid_conversions:,} de {len(df):,}")
    print(f"📊 Tasa de conversión: {conversion_rate:.1f}%")

    if valid_conversions > 0:
        times = df['TimeAlive_seconds'].dropna()
        print(
            f"📈 Rango de tiempos: {times.min():.1f} - {times.max():.1f} segundos")
        print(f"📊 Tiempo promedio: {times.mean():.1f} segundos")

    return df

# Función de limpieza (de tu código XGBoost)


def intelligent_data_cleaning(df):
    """Limpieza inteligente que conserva máximos datos"""
    print("\n🧹 PASO 2: LIMPIEZA INTELIGENTE DE DATOS")
    print("-" * 38)

    initial_count = len(df)
    print(f"📊 Registros iniciales: {initial_count:,}")

    # Eliminar solo registros sin tiempo válido
    df_clean = df.dropna(subset=['TimeAlive_seconds']).copy()
    after_nan = len(df_clean)
    print(
        f"✅ Después de eliminar NaN: {after_nan:,} ({after_nan/initial_count*100:.1f}%)")

    # Filtrar valores negativos
    df_clean = df_clean[df_clean['TimeAlive_seconds'] >= 0]
    after_negative = len(df_clean)
    print(f"✅ Después de eliminar negativos: {after_negative:,}")

    # Usar percentiles muy conservadores para CS:GO
    q001 = df_clean['TimeAlive_seconds'].quantile(0.001)  # 0.1% inferior
    q999 = df_clean['TimeAlive_seconds'].quantile(0.999)  # 0.1% superior

    print(f"🔍 Percentiles: {q001:.1f}s - {q999:.1f}s")

    # Aplicar filtros de extremos
    df_clean = df_clean[
        (df_clean['TimeAlive_seconds'] >= q001) &
        (df_clean['TimeAlive_seconds'] <= q999)
    ]

    final_count = len(df_clean)
    conservation_rate = final_count / initial_count * 100

    print(f"✅ Registros finales: {final_count:,}")
    print(f"📊 Tasa de conservación: {conservation_rate:.1f}%")

    if conservation_rate >= 90:
        print("🎉 Excelente conservación de datos!")
    elif conservation_rate >= 70:
        print("✅ Buena conservación de datos")
    else:
        print("🟡 Conservación moderada")

    return df_clean

# Función de preparación de features (de tu código XGBoost)


def prepare_complete_features(df):
    """Preparar todas las features disponibles para modelado"""
    print("\n⚙️ PASO 3: PREPARACIÓN COMPLETA DE FEATURES")
    print("-" * 41)

    # Crear variable objetivo binaria
    median_time = df['TimeAlive_seconds'].median()
    df['SurvivalClass'] = (df['TimeAlive_seconds'] > median_time).astype(int)

    print(f"🎯 Variable objetivo creada:")
    print(f"   Mediana: {median_time:.1f} segundos")
    balance = df['SurvivalClass'].value_counts(normalize=True)
    print(f"   Balance: Clase 0: {balance[0]:.1%}, Clase 1: {balance[1]:.1%}")

    # Features numéricas candidatas
    numeric_candidates = [
        'MatchKills', 'RoundKills',
        'MatchAssists', 'RoundAssists',
        'MatchHeadshots', 'RoundHeadshots',
        'MatchFlankKills', 'RoundFlankKills',
        'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
        'TravelledDistance', 'FirstKillTime',
        'RLethalGrenadesThrown', 'RNonLethalGrenadesThrown',
        'PrimaryAssaultRifle', 'PrimarySniperRifle', 'PrimaryHeavy',
        'PrimarySMG', 'PrimaryPistol'
    ]

    # Procesar features numéricas
    numeric_features = []
    print(f"\n🔢 Procesando features numéricas:")

    for col in numeric_candidates:
        if col in df.columns:
            # Limpiar formato si es texto
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(
                    '.', '').str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Rellenar NaN y verificar varianza
            df[col] = df[col].fillna(0)

            if df[col].var() > 0:
                numeric_features.append(col)
                print(f"   ✅ {col}")

    # Crear features derivadas específicas para CS:GO
    derived_features = []
    print(f"\n🔧 Creando features derivadas:")

    # KD Ratio (Kill/Death)
    if 'MatchKills' in numeric_features:
        df['KD_Ratio'] = df['MatchKills'] / (1 - df['SurvivalClass'] + 0.1)
        derived_features.append('KD_Ratio')
        print(f"   ✅ KD_Ratio")

    # Eficiencia de headshots
    if 'MatchHeadshots' in numeric_features and 'MatchKills' in numeric_features:
        df['Headshot_Efficiency'] = df['MatchHeadshots'] / \
            (df['MatchKills'] + 1)
        derived_features.append('Headshot_Efficiency')
        print(f"   ✅ Headshot_Efficiency")

    # ROI de equipamiento
    if 'RoundStartingEquipmentValue' in numeric_features and 'MatchKills' in numeric_features:
        df['Equipment_ROI'] = df['MatchKills'] / \
            (df['RoundStartingEquipmentValue'] / 1000 + 1)
        derived_features.append('Equipment_ROI')
        print(f"   ✅ Equipment_ROI")

    # Ratio de asistencias
    if 'MatchAssists' in numeric_features and 'MatchKills' in numeric_features:
        df['Assist_Ratio'] = df['MatchAssists'] / \
            (df['MatchKills'] + df['MatchAssists'] + 1)
        derived_features.append('Assist_Ratio')
        print(f"   ✅ Assist_Ratio")

    # Codificar variables categóricas
    categorical_features = []
    print(f"\n🏷️ Codificando variables categóricas:")

    if 'Map' in df.columns:
        le_map = LabelEncoder()
        df['Map_Encoded'] = le_map.fit_transform(df['Map'].fillna('Unknown'))
        categorical_features.append('Map_Encoded')
        print(f"   ✅ Map_Encoded: {df['Map'].nunique()} mapas")

    if 'Team' in df.columns:
        le_team = LabelEncoder()
        df['Team_Encoded'] = le_team.fit_transform(
            df['Team'].fillna('Unknown'))
        categorical_features.append('Team_Encoded')
        print(f"   ✅ Team_Encoded: {df['Team'].nunique()} equipos")

    # Limpiar features derivadas
    for feat in derived_features:
        df[feat] = df[feat].replace([np.inf, -np.inf], 0).fillna(0)

    # Combinar todas las features
    all_features = numeric_features + derived_features + categorical_features

    print(f"\n📊 Resumen de features:")
    print(f"   Numéricas originales: {len(numeric_features)}")
    print(f"   Derivadas: {len(derived_features)}")
    print(f"   Categóricas: {len(categorical_features)}")
    print(f"   TOTAL: {len(all_features)}")

    return df, all_features


# Ejecutar pipeline XGBoost
print("🚀 Ejecutando pipeline XGBoost...")

# PASO 1-3: Cargar, limpiar y preparar datos
df = load_and_convert_csgo_data()
df_clean = intelligent_data_cleaning(df)
df_prepared, xgb_features = prepare_complete_features(df_clean)

# PASO 4: Entrenar XGBoost
print("\n🤖 PASO 4: ENTRENAMIENTO XGBOOST")
print("-" * 30)

# Preparar datos para XGBoost
X_xgb = df_prepared[xgb_features]
y_xgb = df_prepared['SurvivalClass']

# División estratificada
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

print(f"📊 División XGBoost:")
print(f"   Training: {X_train_xgb.shape}")
print(f"   Test: {X_test_xgb.shape}")
print(f"   Features: {len(xgb_features)}")

# Entrenar XGBoost
print(f"\n🚀 Entrenando XGBoost...")

# Calcular balance de clases
scale_pos_weight = len(
    y_train_xgb[y_train_xgb == 0]) / len(y_train_xgb[y_train_xgb == 1])

# Parámetros específicos para CS:GO
xgb_params = {
    'n_estimators': 300,
    'learning_rate': 0.1,
    'max_depth': 8,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'min_child_weight': 3,
    'gamma': 0.1,
    'scale_pos_weight': scale_pos_weight,
    'random_state': 42,
    'eval_metric': 'auc',
    'verbosity': 0,
    'n_jobs': -1
}

xgb_model = xgb.XGBClassifier(**xgb_params)
xgb_model.fit(X_train_xgb, y_train_xgb)

# Evaluar XGBoost
xgb_acc = xgb_model.score(X_test_xgb, y_test_xgb)
xgb_auc = roc_auc_score(y_test_xgb, xgb_model.predict_proba(X_test_xgb)[:, 1])

print(f"✅ XGBoost entrenado:")
print(f"   Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
print(f"   AUC: {xgb_auc:.4f}")

# =============================================================================
# PARTE 2: PREPARAR DATOS PARA RANDOM FOREST (REGRESIÓN)
# =============================================================================

print("\n" + "="*60)
print("PARTE 2: ENTRENANDO RANDOM FOREST (REGRESIÓN)")
print("="*60)

# Usar mismo dataset pero preparar para regresión
# Variables globales necesarias para el código Random Forest
TARGET_REGRESSION = 'TimeAlive'
FEATURES = [
    'TeamStartingEquipmentValue', 'Map_Encoded', 'EquipmentROI',
    'RoundStartingEquipmentValue', 'HeadshotEfficiency', 'MatchKills',
    'AssistEfficiency', 'MatchHeadshots', 'MatchAssists', 'Team_Encoded'
]

# Preparar datos para Random Forest
print("📊 Preparando datos para Random Forest...")

# Usar TimeAlive_seconds como target (no la clase binaria)
df_rf = df_prepared.copy()
df_rf[TARGET_REGRESSION] = df_rf['TimeAlive_seconds']

# Crear las características específicas que usa tu Random Forest
df_rf['EquipmentROI'] = df_rf['Equipment_ROI']
df_rf['HeadshotEfficiency'] = df_rf['Headshot_Efficiency']
df_rf['AssistEfficiency'] = df_rf['Assist_Ratio']

# Verificar que todas las features existen
available_features = []
for feat in FEATURES:
    if feat in df_rf.columns:
        available_features.append(feat)
        print(f"   ✅ {feat}")
    else:
        print(f"   ❌ {feat} - no disponible")

# Actualizar FEATURES con las disponibles
FEATURES = available_features

# Preparar datos train/test para RF
# Filtrar datos válidos para regresión
df_rf_valid = df_rf.dropna(subset=[TARGET_REGRESSION] + FEATURES)
# Solo tiempos positivos
df_rf_valid = df_rf_valid[df_rf_valid[TARGET_REGRESSION] > 0]

print(f"📊 Datos válidos para RF: {len(df_rf_valid):,}")

# Variables globales necesarias para tu función Random Forest
X_all = df_rf_valid[FEATURES + [TARGET_REGRESSION]]
y_all = df_rf_valid[TARGET_REGRESSION]

X_train, X_test, y_train, y_test = train_test_split(
    df_rf_valid[FEATURES], df_rf_valid[TARGET_REGRESSION],
    test_size=0.2, random_state=42
)

print(f"📊 División Random Forest:")
print(f"   Training: {X_train.shape}")
print(f"   Test: {X_test.shape}")
print(f"   Features: {len(FEATURES)}")

# Ejecutar tu función Random Forest


def random_forest_model():
    """Random Forest con búsqueda aleatoria de hiperparámetros"""

    print(f"\n🌲 CONFIGURACIÓN RANDOM FOREST:")
    print(f"Variable objetivo: {TARGET_REGRESSION}")
    print(f"Features utilizadas: {len(FEATURES)}")
    print(f"Datos de entrenamiento: {len(X_train):,} registros")
    print(f"Datos de prueba: {len(X_test):,} registros")

    # Parámetros para búsqueda aleatoria (adaptados para dataset pequeño)
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    print(f"\n⚙️ PARÁMETROS PARA BÚSQUEDA ALEATORIA:")
    for param, values in param_distributions.items():
        print(f"   {param}: {values}")

    print(f"\n🔍 REALIZANDO BÚSQUEDA ALEATORIA...")
    print(f"Esto puede tomar unos minutos...")

    # Modelo base
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Búsqueda aleatoria (adaptada para dataset pequeño)
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=15,  # Reducir iteraciones
        cv=3,       # 3-fold CV
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    # Entrenar con búsqueda de hiperparámetros
    rf_random.fit(X_train[FEATURES], y_train)

    # Mejor modelo encontrado
    best_rf = rf_random.best_estimator_

    print(f"\n🏆 MEJORES HIPERPARÁMETROS ENCONTRADOS:")
    for param, value in rf_random.best_params_.items():
        print(f"   {param}: {value}")

    print(f"\nMejor score de validación cruzada: {rf_random.best_score_:.4f}")

    # Predicciones con el mejor modelo
    y_pred_train = best_rf.predict(X_train[FEATURES])
    y_pred_test = best_rf.predict(X_test[FEATURES])

    # Métricas del modelo optimizado
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"\n📊 RESULTADOS RANDOM FOREST OPTIMIZADO:")
    print(f"R² Train: {r2_train:.4f}")
    print(f"R² Test: {r2_test:.4f}")
    print(f"RMSE Test: {rmse_test:.4f}")
    print(f"MAE Test: {mae_test:.4f}")

    return best_rf, rf_random.best_params_, r2_test, rmse_test, mae_test


# Ejecutar Random Forest
print("🌳 Ejecutando Random Forest...")
results_rf = random_forest_model()

# =============================================================================
# PARTE 3: EXPORTAR MODELOS REALES
# =============================================================================

print("\n" + "="*60)
print("PARTE 3: EXPORTANDO MODELOS REALES")
print("="*60)

# Crear carpeta models
os.makedirs('models', exist_ok=True)

# Exportar XGBoost
print("📦 Exportando XGBoost...")
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

with open('models/xgboost_features.pkl', 'wb') as f:
    pickle.dump(xgb_features, f)

print(f"✅ XGBoost exportado:")
print(f"   Accuracy: {xgb_acc*100:.2f}%")
print(f"   AUC: {xgb_auc:.4f}")
print(f"   Features: {len(xgb_features)}")

# Exportar Random Forest
print("\n📦 Exportando Random Forest...")
rf_model = results_rf[0]
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('models/rf_features.pkl', 'wb') as f:
    pickle.dump(FEATURES, f)

print(f"✅ Random Forest exportado:")
print(f"   R²: {results_rf[2]:.4f}")
print(f"   RMSE: {results_rf[3]:.4f}")
print(f"   Features: {len(FEATURES)}")

# Crear archivo con metadatos
metadata = {
    'xgboost': {
        'accuracy': xgb_acc,
        'auc': xgb_auc,
        'features': xgb_features,
        'model_type': 'classification'
    },
    'random_forest': {
        'r2': results_rf[2],
        'rmse': results_rf[3],
        'mae': results_rf[4],
        'features': FEATURES,
        'model_type': 'regression'
    },
    'dataset': 'Anexo ET_demo_round_traces_2022 (1).csv',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

with open('models/model_metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

# Verificación final
print("\n🧪 VERIFICACIÓN FINAL:")
for filename in ['xgboost_model.pkl', 'random_forest_model.pkl',
                 'xgboost_features.pkl', 'rf_features.pkl', 'model_metadata.pkl']:
    filepath = os.path.join('models', filename)
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"   ✅ {filename} - válido ({type(data).__name__})")
    except Exception as e:
        print(f"   ❌ {filename} - error: {e}")

print("\n🎉 MODELOS REALES EXPORTADOS EXITOSAMENTE!")
print("📊 Usando tu dataset original: Anexo ET_demo_round_traces_2022 (1).csv")
print("🚀 Ahora ejecuta: python app.py")

print(f"\n📋 RESUMEN:")
print(f"🎯 XGBoost (Clasificación): {xgb_acc*100:.2f}% accuracy")
print(f"🌳 Random Forest (Regresión): R² = {results_rf[2]:.4f}")
print(f"📁 Archivos exportados en: models/")
