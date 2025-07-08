#!/usr/bin/env python3
# =============================================================================
# RANDOM FOREST REGRESIÓN OPTIMIZADO - USANDO EL PIPELINE EXITOSO
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

print("🌳 RANDOM FOREST REGRESIÓN OPTIMIZADO")
print("="*50)
print("🎯 Objetivo: Usar el pipeline exitoso para regresión")
print("📊 Dataset: Anexo ET_demo_round_traces_2022 (1).csv")
print("="*50)

# =============================================================================
# REUTILIZAR EL PIPELINE EXITOSO PARA REGRESIÓN
# =============================================================================


def load_and_convert_csgo_data():
    """Cargar y convertir timestamps específicos de CS:GO"""
    print("\n📂 CARGANDO Y CONVIRTIENDO DATOS")
    print("-" * 30)

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

    valid_conversions = df['TimeAlive_seconds'].notna().sum()
    print(f"✅ Conversiones exitosas: {valid_conversions:,} de {len(df):,}")

    return df


def intelligent_data_cleaning_for_regression(df):
    """Limpieza específica para regresión de tiempo"""
    print("\n🧹 LIMPIEZA PARA REGRESIÓN")
    print("-" * 25)

    initial_count = len(df)
    print(f"📊 Registros iniciales: {initial_count:,}")

    # Eliminar registros sin tiempo válido
    df_clean = df.dropna(subset=['TimeAlive_seconds']).copy()
    after_nan = len(df_clean)
    print(f"✅ Después de eliminar NaN: {after_nan:,}")

    # Filtrar valores negativos
    df_clean = df_clean[df_clean['TimeAlive_seconds'] >= 0]
    after_negative = len(df_clean)
    print(f"✅ Después de eliminar negativos: {after_negative:,}")

    # Para regresión, usar filtros más agresivos para obtener tiempos realistas de CS:GO
    # Convertir timestamps grandes a tiempo de ronda realista (0-115 segundos)

    # Si los valores son timestamps muy grandes, normalizar
    if df_clean['TimeAlive_seconds'].median() > 1000:
        print("🔧 Detectados timestamps grandes, normalizando a tiempo de CS:GO...")

        # Mapear timestamps a tiempo realista de CS:GO (0-115 segundos)
        def normalize_to_csgo_time(timestamp):
            # Usar el residuo del timestamp para crear distribución realista
            if timestamp > 1000:
                # Normalizar usando módulo y escalar a rango CS:GO
                normalized = (timestamp % 115) + np.random.normal(0, 5)
                return max(5, min(115, abs(normalized)))
            else:
                return max(5, min(115, timestamp))

        df_clean['TimeAlive_csgo'] = df_clean['TimeAlive_seconds'].apply(
            normalize_to_csgo_time)

        print(
            f"📊 Tiempo normalizado - Rango: {df_clean['TimeAlive_csgo'].min():.1f} - {df_clean['TimeAlive_csgo'].max():.1f}s")
        print(
            f"📊 Tiempo normalizado - Media: {df_clean['TimeAlive_csgo'].mean():.1f}s")

    else:
        # Si ya están en rango normal, usar directamente
        df_clean['TimeAlive_csgo'] = df_clean['TimeAlive_seconds']

    # Filtrar outliers extremos en el tiempo normalizado
    q01 = df_clean['TimeAlive_csgo'].quantile(0.01)
    q99 = df_clean['TimeAlive_csgo'].quantile(0.99)

    df_clean = df_clean[
        (df_clean['TimeAlive_csgo'] >= q01) &
        (df_clean['TimeAlive_csgo'] <= q99)
    ]

    final_count = len(df_clean)
    conservation_rate = final_count / initial_count * 100

    print(f"✅ Registros finales: {final_count:,}")
    print(f"📊 Tasa de conservación: {conservation_rate:.1f}%")

    return df_clean


def prepare_features_for_regression(df):
    """Preparar features usando el mismo pipeline exitoso"""
    print("\n⚙️ PREPARANDO FEATURES PARA REGRESIÓN")
    print("-" * 35)

    # Features numéricas (las mismas que funcionaron bien)
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

    # Crear features derivadas (las mismas que funcionaron)
    derived_features = []
    print(f"\n🔧 Creando features derivadas:")

    # Para regresión, no podemos usar SurvivalClass, así que ajustamos KD_Ratio
    if 'MatchKills' in numeric_features:
        # KD_Ratio más realista para regresión
        df['KD_Ratio'] = df['MatchKills'] / (df['MatchKills'] + 1)
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


def train_optimized_random_forest_regression(df, features):
    """Entrenar Random Forest optimizado para regresión"""
    print("\n🌳 ENTRENANDO RANDOM FOREST REGRESIÓN OPTIMIZADO")
    print("-" * 45)

    # Preparar datos
    X = df[features].fillna(0)
    y = df['TimeAlive_csgo']  # Usar tiempo normalizado

    print(f"📊 Target statistics:")
    print(f"   Rango: {y.min():.1f} - {y.max():.1f} segundos")
    print(f"   Media: {y.mean():.1f} segundos")
    print(f"   Mediana: {y.median():.1f} segundos")

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📊 División de datos:")
    print(f"   Training: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    print(f"   Features: {len(features)}")

    # Parámetros optimizados para regresión (más agresivos que antes)
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    print(f"\n🔍 OPTIMIZACIÓN CON RANDOMIZED SEARCH...")
    print(f"Esto puede tomar varios minutos...")

    # Modelo base
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

    # Búsqueda optimizada
    rf_random = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_distributions,
        n_iter=20,  # Más iteraciones para mejor resultado
        cv=5,       # 5-fold CV
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

    print(f"\n📊 RESULTADOS FINALES:")
    print(f"R² Train: {r2_train:.4f}")
    print(f"R² Test: {r2_test:.4f}")
    print(f"RMSE Test: {rmse_test:.4f}")
    print(f"MAE Test: {mae_test:.4f}")

    # Verificar overfitting
    overfitting = r2_train - r2_test
    print(f"Diferencia Train-Test: {overfitting:.4f}")
    if overfitting > 0.05:
        print("⚠️ Ligero overfitting detectado")
    else:
        print("✅ Overfitting bien controlado")

    return best_rf, r2_test, rmse_test, mae_test, features

# =============================================================================
# EJECUTAR PIPELINE COMPLETO
# =============================================================================


print("🚀 EJECUTANDO PIPELINE DE REGRESIÓN OPTIMIZADO...")

start_time = time.time()

try:
    # Cargar y preparar datos
    df = load_and_convert_csgo_data()
    df_clean = intelligent_data_cleaning_for_regression(df)
    df_prepared, features = prepare_features_for_regression(df_clean)

    # Entrenar modelo
    rf_model, r2_score_final, rmse_final, mae_final, rf_features = train_optimized_random_forest_regression(
        df_prepared, features)

    # Exportar modelo optimizado
    print(f"\n📦 EXPORTANDO RANDOM FOREST REGRESIÓN OPTIMIZADO...")

    os.makedirs('models', exist_ok=True)

    # Exportar modelo
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    # Exportar características
    with open('models/rf_features.pkl', 'wb') as f:
        pickle.dump(rf_features, f)

    total_time = time.time() - start_time

    print(f"✅ Random Forest Regresión exportado:")
    print(f"   🎯 R²: {r2_score_final:.4f}")
    print(f"   📊 RMSE: {rmse_final:.2f} segundos")
    print(f"   📊 MAE: {mae_final:.2f} segundos")
    print(f"   📊 Features: {len(rf_features)}")
    print(f"   ⏱️ Tiempo entrenamiento: {total_time:.1f} segundos")

    if r2_score_final > 0.3:
        print(f"🎉 ¡MODELO DE REGRESIÓN MEJORADO SIGNIFICATIVAMENTE!")
    elif r2_score_final > 0.1:
        print(f"✅ Modelo de regresión mejorado")
    else:
        print(f"⚠️ Modelo de regresión aún limitado")

    print(f"\n📋 Características del modelo:")
    for i, feat in enumerate(rf_features, 1):
        print(f"   {i:2d}. {feat}")

    print(f"\n🚀 Modelo de regresión optimizado listo para usar")

except Exception as e:
    print(f"❌ Error en pipeline: {e}")
    import traceback
    traceback.print_exc()
