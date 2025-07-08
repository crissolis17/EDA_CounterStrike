#!/usr/bin/env python3
# =============================================================================
# SCRIPT SIMPLIFICADO PARA ENTRENAR Y EXPORTAR MODELOS REALES
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
import time

# Importaciones básicas que deberían funcionar
try:
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    print("✅ Sklearn básico importado")
except ImportError as e:
    print(f"❌ Error en sklearn: {e}")
    exit(1)

try:
    import xgboost as xgb
    print("✅ XGBoost disponible")
except ImportError:
    print("📦 Instalando XGBoost...")
    import subprocess
    subprocess.run(["pip", "install", "xgboost"])
    try:
        import xgboost as xgb
        print("✅ XGBoost instalado e importado")
    except ImportError as e:
        print(f"❌ No se pudo instalar XGBoost: {e}")
        exit(1)

warnings.filterwarnings('ignore')

print("🚀 ENTRENAMIENTO SIMPLIFICADO DE MODELOS REALES")
print("="*60)

# Verificar dataset
dataset_path = 'Anexo ET_demo_round_traces_2022 (1).csv'
if not os.path.exists(dataset_path):
    print(f"❌ No se encuentra: {dataset_path}")
    print("📁 Copia el archivo a la carpeta backend/")
    exit(1)

print(f"✅ Dataset encontrado: {dataset_path}")

# =============================================================================
# PARTE 1: CARGAR Y PREPARAR DATOS
# =============================================================================

print("\n📂 CARGANDO DATOS...")
try:
    df = pd.read_csv(dataset_path, sep=';')
    print(
        f"✅ Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    exit(1)

# Función simplificada para convertir timestamps


def convert_timestamp_simple(timestamp_str):
    """Conversión simplificada de timestamps"""
    if pd.isna(timestamp_str):
        return np.nan
    try:
        cleaned = str(timestamp_str).replace('.', '')
        timestamp_num = float(cleaned)

        if timestamp_num > 1e15:
            seconds = timestamp_num / 1e9
        elif timestamp_num > 1e12:
            seconds = timestamp_num / 1e6
        elif timestamp_num > 1e9:
            seconds = timestamp_num / 1e3
        else:
            seconds = timestamp_num

        return seconds
    except:
        return np.nan


print("🔄 Convirtiendo timestamps...")
df['TimeAlive_seconds'] = df['TimeAlive'].apply(convert_timestamp_simple)

# Limpiar datos básico
df_clean = df.dropna(subset=['TimeAlive_seconds']).copy()
df_clean = df_clean[df_clean['TimeAlive_seconds'] >= 0]

# Filtros básicos por percentiles
q01 = df_clean['TimeAlive_seconds'].quantile(0.01)
q99 = df_clean['TimeAlive_seconds'].quantile(0.99)
df_clean = df_clean[
    (df_clean['TimeAlive_seconds'] >= q01) &
    (df_clean['TimeAlive_seconds'] <= q99)
]

print(f"✅ Datos limpiados: {len(df_clean):,} registros")

# =============================================================================
# PARTE 2: PREPARAR FEATURES BÁSICAS
# =============================================================================

print("\n🔧 PREPARANDO CARACTERÍSTICAS...")

# Features numéricas básicas disponibles
numeric_features = []
candidates = [
    'MatchKills', 'RoundKills', 'MatchAssists', 'RoundAssists',
    'MatchHeadshots', 'RoundHeadshots', 'RoundStartingEquipmentValue',
    'TeamStartingEquipmentValue', 'TravelledDistance'
]

for col in candidates:
    if col in df_clean.columns:
        try:
            if df_clean[col].dtype == 'object':
                df_clean[col] = pd.to_numeric(df_clean[col].astype(str).str.replace(
                    '.', '').str.replace(',', '.'), errors='coerce')
            df_clean[col] = df_clean[col].fillna(0)
            if df_clean[col].var() > 0:
                numeric_features.append(col)
                print(f"   ✅ {col}")
        except:
            print(f"   ❌ {col} - error en procesamiento")

# Variables categóricas básicas
categorical_features = []
if 'Map' in df_clean.columns:
    try:
        le_map = LabelEncoder()
        df_clean['Map_Encoded'] = le_map.fit_transform(
            df_clean['Map'].fillna('Unknown'))
        categorical_features.append('Map_Encoded')
        print(f"   ✅ Map_Encoded")
    except:
        print(f"   ❌ Map_Encoded - error")

if 'Team' in df_clean.columns:
    try:
        le_team = LabelEncoder()
        df_clean['Team_Encoded'] = le_team.fit_transform(
            df_clean['Team'].fillna('Unknown'))
        categorical_features.append('Team_Encoded')
        print(f"   ✅ Team_Encoded")
    except:
        print(f"   ❌ Team_Encoded - error")

# Features derivadas básicas
derived_features = []
try:
    if 'MatchKills' in numeric_features:
        df_clean['KD_Ratio'] = df_clean['MatchKills'] / \
            (df_clean['MatchKills'] + 1)
        derived_features.append('KD_Ratio')
        print(f"   ✅ KD_Ratio")
except:
    print(f"   ❌ KD_Ratio - error")

try:
    if 'MatchHeadshots' in numeric_features and 'MatchKills' in numeric_features:
        df_clean['Headshot_Efficiency'] = df_clean['MatchHeadshots'] / \
            (df_clean['MatchKills'] + 1)
        derived_features.append('Headshot_Efficiency')
        print(f"   ✅ Headshot_Efficiency")
except:
    print(f"   ❌ Headshot_Efficiency - error")

# Combinar todas las features
all_features = numeric_features + categorical_features + derived_features
print(f"\n📊 Total features: {len(all_features)}")

# =============================================================================
# PARTE 3: ENTRENAR XGBOOST (CLASIFICACIÓN)
# =============================================================================

print("\n🚀 ENTRENANDO XGBOOST (CLASIFICACIÓN)...")

# Crear variable objetivo binaria
median_time = df_clean['TimeAlive_seconds'].median()
df_clean['SurvivalClass'] = (
    df_clean['TimeAlive_seconds'] > median_time).astype(int)

print(f"🎯 Mediana: {median_time:.1f}s")
print(f"🎯 Balance: {df_clean['SurvivalClass'].value_counts().to_dict()}")

# Preparar datos para XGBoost
X_xgb = df_clean[all_features].fillna(0)
y_xgb = df_clean['SurvivalClass']

# División
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_xgb, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

print(f"📊 XGBoost - Train: {X_train_xgb.shape}, Test: {X_test_xgb.shape}")

# Entrenar XGBoost con parámetros básicos
try:
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    )

    xgb_model.fit(X_train_xgb, y_train_xgb)

    xgb_acc = xgb_model.score(X_test_xgb, y_test_xgb)
    print(
        f"✅ XGBoost entrenado - Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

except Exception as e:
    print(f"❌ Error entrenando XGBoost: {e}")
    xgb_model = None
    xgb_acc = 0

# =============================================================================
# PARTE 4: ENTRENAR RANDOM FOREST (REGRESIÓN)
# =============================================================================

print("\n🌳 ENTRENANDO RANDOM FOREST (REGRESIÓN)...")

# Preparar datos para Random Forest - usar solo features principales
rf_features = []
rf_feature_candidates = [
    'TeamStartingEquipmentValue', 'RoundStartingEquipmentValue',
    'MatchKills', 'MatchHeadshots', 'MatchAssists'
]

for feat in rf_feature_candidates:
    if feat in all_features:
        rf_features.append(feat)

# Agregar features categóricas si están disponibles
if 'Map_Encoded' in all_features:
    rf_features.append('Map_Encoded')
if 'Team_Encoded' in all_features:
    rf_features.append('Team_Encoded')

# Agregar features derivadas si están disponibles
if 'Headshot_Efficiency' in all_features:
    rf_features.append('Headshot_Efficiency')

print(f"📊 RF Features ({len(rf_features)}): {rf_features}")

# Preparar datos para Random Forest
df_rf = df_clean.dropna(subset=['TimeAlive_seconds'] + rf_features)
df_rf = df_rf[df_rf['TimeAlive_seconds'] > 0]

X_rf = df_rf[rf_features].fillna(0)
y_rf = df_rf['TimeAlive_seconds']

# División
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)

print(f"📊 Random Forest - Train: {X_train_rf.shape}, Test: {X_test_rf.shape}")

# Entrenar Random Forest con parámetros básicos
try:
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train_rf, y_train_rf)

    # Evaluar
    y_pred_rf = rf_model.predict(X_test_rf)
    rf_r2 = r2_score(y_test_rf, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))

    print(f"✅ Random Forest entrenado - R²: {rf_r2:.4f}, RMSE: {rf_rmse:.2f}")

except Exception as e:
    print(f"❌ Error entrenando Random Forest: {e}")
    rf_model = None
    rf_r2 = 0

# =============================================================================
# PARTE 5: EXPORTAR MODELOS
# =============================================================================

print("\n📦 EXPORTANDO MODELOS...")

# Crear carpeta
os.makedirs('models', exist_ok=True)

export_count = 0

# Exportar XGBoost
if xgb_model is not None:
    try:
        with open('models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)

        with open('models/xgboost_features.pkl', 'wb') as f:
            pickle.dump(all_features, f)

        print(f"✅ XGBoost exportado - Accuracy: {xgb_acc*100:.2f}%")
        export_count += 1
    except Exception as e:
        print(f"❌ Error exportando XGBoost: {e}")

# Exportar Random Forest
if rf_model is not None:
    try:
        with open('models/random_forest_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)

        with open('models/rf_features.pkl', 'wb') as f:
            pickle.dump(rf_features, f)

        print(f"✅ Random Forest exportado - R²: {rf_r2:.4f}")
        export_count += 1
    except Exception as e:
        print(f"❌ Error exportando Random Forest: {e}")

# Exportar feature names para compatibilidad
try:
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(all_features, f)
    print(f"✅ Feature names exportado")
    export_count += 1
except Exception as e:
    print(f"❌ Error exportando feature names: {e}")

# Verificación final
print(f"\n🧪 VERIFICACIÓN FINAL:")
files_to_check = ['xgboost_model.pkl',
                  'random_forest_model.pkl', 'feature_names.pkl']
verified = 0

for filename in files_to_check:
    filepath = os.path.join('models', filename)
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"   ✅ {filename} - válido")
            verified += 1
        except Exception as e:
            print(f"   ❌ {filename} - error: {e}")
    else:
        print(f"   ❌ {filename} - no existe")

if verified >= 2:
    print(f"\n🎉 EXPORTACIÓN EXITOSA!")
    print(f"📊 {verified}/3 archivos exportados correctamente")
    print(f"🚀 Ahora ejecuta: python app.py")

    if xgb_model is not None:
        print(f"🎯 XGBoost: {xgb_acc*100:.2f}% accuracy")
    if rf_model is not None:
        print(f"🌳 Random Forest: R² = {rf_r2:.4f}")
else:
    print(f"\n❌ FALLÓ LA EXPORTACIÓN")
    print(f"📊 Solo {verified}/3 archivos válidos")
    print(f"🔧 Revisa los errores anteriores")
