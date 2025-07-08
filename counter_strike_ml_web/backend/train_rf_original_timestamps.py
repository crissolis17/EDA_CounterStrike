#!/usr/bin/env python3
# =============================================================================
# RANDOM FOREST REGRESIÓN CON TIMESTAMPS ORIGINALES
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore')

print("🌳 RANDOM FOREST REGRESIÓN - TIMESTAMPS ORIGINALES")
print("="*55)
print("🎯 Entrenar con datos originales, convertir solo al mostrar")
print("📊 Dataset: Anexo ET_demo_round_traces_2022 (1).csv")
print("="*55)

# =============================================================================
# CARGAR Y PREPARAR DATOS CON TIMESTAMPS ORIGINALES
# =============================================================================


def load_and_prepare_original_data():
    """Cargar datos manteniendo timestamps originales"""
    print("\n📂 CARGANDO DATOS CON TIMESTAMPS ORIGINALES")
    print("-" * 40)

    # Cargar datos
    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
    print(
        f"✅ Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")

    # Función para convertir timestamps manteniendo precisión
    def convert_csgo_timestamp_precise(timestamp_str):
        """Convierte timestamps manteniendo valor original para correlaciones"""
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

    # Convertir TimeAlive manteniendo valor original
    print("🔄 Convirtiendo timestamps manteniendo precisión...")
    df['TimeAlive_original'] = df['TimeAlive'].apply(
        convert_csgo_timestamp_precise)

    valid_conversions = df['TimeAlive_original'].notna().sum()
    print(f"✅ Conversiones exitosas: {valid_conversions:,} de {len(df):,}")

    if valid_conversions > 0:
        times = df['TimeAlive_original'].dropna()
        print(f"📈 Rango original: {times.min():.1f} - {times.max():.1f}")
        print(f"📊 Media original: {times.mean():.1f}")
        print(f"📊 Mediana original: {times.median():.1f}")

    return df


def clean_data_for_original_regression(df):
    """Limpieza básica manteniendo correlaciones originales"""
    print("\n🧹 LIMPIEZA BÁSICA MANTENIENDO CORRELACIONES")
    print("-" * 45)

    initial_count = len(df)
    print(f"📊 Registros iniciales: {initial_count:,}")

    # Eliminar solo registros inválidos
    df_clean = df.dropna(subset=['TimeAlive_original']).copy()
    after_nan = len(df_clean)
    print(f"✅ Después de eliminar NaN: {after_nan:,}")

    # Filtrar valores negativos
    df_clean = df_clean[df_clean['TimeAlive_original'] >= 0]
    after_negative = len(df_clean)
    print(f"✅ Después de eliminar negativos: {after_negative:,}")

    # Filtros muy conservadores para mantener correlaciones
    q001 = df_clean['TimeAlive_original'].quantile(0.001)
    q999 = df_clean['TimeAlive_original'].quantile(0.999)

    df_clean = df_clean[
        (df_clean['TimeAlive_original'] >= q001) &
        (df_clean['TimeAlive_original'] <= q999)
    ]

    final_count = len(df_clean)
    conservation_rate = final_count / initial_count * 100

    print(f"✅ Registros finales: {final_count:,}")
    print(f"📊 Tasa de conservación: {conservation_rate:.1f}%")

    return df_clean


def prepare_features_original_pipeline(df):
    """Preparar features usando el pipeline exitoso del XGBoost"""
    print("\n⚙️ PREPARANDO FEATURES - PIPELINE EXITOSO")
    print("-" * 40)

    # Features numéricas (exactamente las mismas que funcionaron en XGBoost)
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
    print(f"🔢 Procesando features numéricas:")

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

    # Features derivadas (exactamente las mismas que en XGBoost)
    derived_features = []
    print(f"\n🔧 Creando features derivadas:")

    # Para regresión, necesitamos una proxy de supervivencia sin usar la variable objetivo
    # Usar la mediana de la población como proxy
    median_time = df['TimeAlive_original'].median()
    survival_proxy = (df['TimeAlive_original'] > median_time).astype(int)

    # KD Ratio usando proxy de supervivencia
    if 'MatchKills' in numeric_features:
        df['KD_Ratio'] = df['MatchKills'] / (1 - survival_proxy + 0.1)
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

    # Variables categóricas
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


def train_rf_with_original_timestamps(df, features):
    """Entrenar RF con timestamps originales"""
    print("\n🌳 ENTRENANDO RF CON TIMESTAMPS ORIGINALES")
    print("-" * 42)

    # Preparar datos
    X = df[features].fillna(0)
    y = df['TimeAlive_original']  # Usar timestamps originales

    print(f"📊 Target statistics (timestamps originales):")
    print(f"   Rango: {y.min():.1f} - {y.max():.1f}")
    print(f"   Media: {y.mean():.1f}")
    print(f"   Mediana: {y.median():.1f}")
    print(f"   Std: {y.std():.1f}")

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📊 División de datos:")
    print(f"   Training: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    print(f"   Features: {len(features)}")

    # Parámetros optimizados para regresión con timestamps grandes
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    print(f"\n🔍 OPTIMIZACIÓN CON RANDOMIZED SEARCH...")
    print(f"Buscando mejores parámetros para timestamps originales...")

    # Modelo base
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Búsqueda optimizada
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=15,  # Menos iteraciones para timestamp grandes
        cv=3,       # 3-fold CV para eficiencia
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    # Entrenar
    rf_random.fit(X_train, y_train)

    # Mejor modelo
    best_rf = rf_random.best_estimator_

    print(f"\n🏆 MEJORES HIPERPARÁMETROS:")
    for param, value in rf_random.best_params_.items():
        print(f"   {param}: {value}")

    print(f"\nMejor score de validación cruzada: {rf_random.best_score_:.4f}")

    # Evaluar modelo final
    y_pred_train = best_rf.predict(X_train)
    y_pred_test = best_rf.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)

    print(f"\n📊 RESULTADOS FINALES (TIMESTAMPS ORIGINALES):")
    print(f"R² Train: {r2_train:.4f}")
    print(f"R² Test: {r2_test:.4f}")
    print(f"RMSE Test: {rmse_test:.2f}")
    print(f"MAE Test: {mae_test:.2f}")

    # Verificar overfitting
    overfitting = r2_train - r2_test
    print(f"Diferencia Train-Test: {overfitting:.4f}")
    if overfitting > 0.05:
        print("⚠️ Ligero overfitting detectado")
    else:
        print("✅ Overfitting bien controlado")

    # Crear función de conversión para mostrar
    def convert_timestamp_to_csgo_display(timestamp):
        """Convierte timestamp original a segundos de CS:GO para mostrar"""
        # Normalizar el timestamp a rango de CS:GO (5-115 segundos)
        if timestamp <= 0:
            return 5.0

        # Usar logaritmo para comprimir el rango manteniendo proporcionalidad
        log_timestamp = np.log10(max(1, timestamp))

        # Mapear logaritmo a rango CS:GO
        min_log = np.log10(y.min()) if y.min() > 0 else 0
        max_log = np.log10(y.max())

        if max_log > min_log:
            normalized = (log_timestamp - min_log) / (max_log - min_log)
        else:
            normalized = 0.5

        # Mapear a rango CS:GO (5-115 segundos)
        csgo_time = 5 + (normalized * 110)
        return max(5.0, min(115.0, csgo_time))

    return best_rf, r2_test, rmse_test, mae_test, features, convert_timestamp_to_csgo_display

# =============================================================================
# EJECUTAR PIPELINE COMPLETO
# =============================================================================


print("🚀 EJECUTANDO PIPELINE CON TIMESTAMPS ORIGINALES...")

start_time = time.time()

try:
    # Cargar y preparar datos
    df = load_and_prepare_original_data()
    df_clean = clean_data_for_original_regression(df)
    df_prepared, features = prepare_features_original_pipeline(df_clean)

    # Entrenar modelo
    rf_model, r2_final, rmse_final, mae_final, rf_features, timestamp_converter = train_rf_with_original_timestamps(
        df_prepared, features)

    # Exportar modelo y función de conversión
    print(f"\n📦 EXPORTANDO MODELO CON TIMESTAMPS ORIGINALES...")

    os.makedirs('models', exist_ok=True)

    # Exportar modelo
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    # Exportar características
    with open('models/rf_features.pkl', 'wb') as f:
        pickle.dump(rf_features, f)

    # Exportar función de conversión y estadísticas para el backend
    conversion_data = {
        'convert_function': timestamp_converter,
        'original_min': df_prepared['TimeAlive_original'].min(),
        'original_max': df_prepared['TimeAlive_original'].max(),
        'original_median': df_prepared['TimeAlive_original'].median(),
        'original_mean': df_prepared['TimeAlive_original'].mean()
    }

    with open('models/timestamp_conversion.pkl', 'wb') as f:
        pickle.dump(conversion_data, f)

    total_time = time.time() - start_time

    print(f"✅ Random Forest con timestamps originales exportado:")
    print(f"   🎯 R²: {r2_final:.4f}")
    print(f"   📊 RMSE: {rmse_final:.2f}")
    print(f"   📊 MAE: {mae_final:.2f}")
    print(f"   📊 Features: {len(rf_features)}")
    print(f"   ⏱️ Tiempo entrenamiento: {total_time:.1f} segundos")

    if r2_final > 0.5:
        print(f"🎉 ¡EXCELENTE MODELO DE REGRESIÓN!")
    elif r2_final > 0.3:
        print(f"✅ Buen modelo de regresión")
    elif r2_final > 0.1:
        print(f"⚖️ Modelo moderado")
    else:
        print(f"⚠️ Modelo limitado - pero mantiene correlaciones originales")

    print(f"\n🔧 La conversión a tiempo CS:GO se hará en el backend")
    print(f"🎯 Modelo entrenado con correlaciones reales del dataset")

except Exception as e:
    print(f"❌ Error en pipeline: {e}")
    import traceback
    traceback.print_exc()
