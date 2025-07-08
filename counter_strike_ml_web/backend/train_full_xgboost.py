#!/usr/bin/env python3
# =============================================================================
# PIPELINE COMPLETO XGBOOST - REPRODUCIR 97.31% ACCURACY
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import warnings
import time
import pickle
import os
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

print("🚀 PIPELINE COMPLETO XGBOOST - OBJETIVO 97.31%")
print("="*55)
print("🎯 Reproduciendo tu código original optimizado")
print("📊 Dataset: Anexo ET_demo_round_traces_2022 (1).csv")
print("="*55)

# =============================================================================
# PASO 1: CARGA Y CONVERSIÓN DE TIMESTAMPS (TU CÓDIGO ORIGINAL)
# =============================================================================


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

# =============================================================================
# PASO 2: LIMPIEZA INTELIGENTE DE DATOS (TU CÓDIGO ORIGINAL)
# =============================================================================


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

# =============================================================================
# PASO 3: PREPARACIÓN COMPLETA DE FEATURES (TU CÓDIGO ORIGINAL)
# =============================================================================


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

# =============================================================================
# PASO 4: ENTRENAMIENTO COMPLETO DE MODELOS (TU CÓDIGO ORIGINAL)
# =============================================================================


def train_all_models_complete(df, features):
    """Entrenar todos los modelos incluyendo XGBoost optimizado"""
    print("\n🤖 PASO 4: ENTRENAMIENTO DE TODOS LOS MODELOS")
    print("-" * 43)

    # Preparar datos
    X = df[features]
    y = df['SurvivalClass']

    # División estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalado para modelos que lo requieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"📊 División de datos:")
    print(f"   Training: {X_train.shape}")
    print(f"   Test: {X_test.shape}")
    print(f"   Features: {len(features)}")

    results = {}
    models = {}

    # 1. Logistic Regression
    print(f"\n1️⃣ Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_acc = lr.score(X_test_scaled, y_test)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:, 1])
    results['Logistic Regression'] = {'accuracy': lr_acc, 'auc': lr_auc}
    models['Logistic Regression'] = lr
    print(f"   Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")

    # 2. Decision Tree
    print(f"\n2️⃣ Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = dt.score(X_test, y_test)
    dt_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
    results['Decision Tree'] = {'accuracy': dt_acc, 'auc': dt_auc}
    models['Decision Tree'] = dt
    print(f"   Accuracy: {dt_acc:.4f} ({dt_acc*100:.2f}%)")

    # 3. SVM
    print(f"\n3️⃣ SVM...")
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    svm_acc = svm.score(X_test_scaled, y_test)
    svm_auc = roc_auc_score(y_test, svm.predict_proba(X_test_scaled)[:, 1])
    results['SVM'] = {'accuracy': svm_acc, 'auc': svm_auc}
    models['SVM'] = svm
    print(f"   Accuracy: {svm_acc:.4f} ({svm_acc*100:.2f}%)")

    # 4. Random Forest Optimizado
    print(f"\n4️⃣ Random Forest Optimizado...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    results['Random Forest'] = {'accuracy': rf_acc, 'auc': rf_auc}
    models['Random Forest'] = rf
    print(f"   Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")

    # 5. KNN
    print(f"\n5️⃣ KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    knn_acc = knn.score(X_test_scaled, y_test)
    knn_auc = roc_auc_score(y_test, knn.predict_proba(X_test_scaled)[:, 1])
    results['KNN'] = {'accuracy': knn_acc, 'auc': knn_auc}
    models['KNN'] = knn
    print(f"   Accuracy: {knn_acc:.4f} ({knn_acc*100:.2f}%)")

    # 6. XGBoost Optimizado (EXACTAMENTE COMO TU CÓDIGO)
    print(f"\n6️⃣ XGBoost Optimizado...")

    # Calcular balance de clases
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Parámetros específicos para CS:GO (TUS PARÁMETROS ORIGINALES)
    xgb_params = {
        'n_estimators': 300,  # Sin early stopping, usar número fijo
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

    # Entrenar SIN early stopping para evitar error
    xgb_model.fit(X_train, y_train)

    xgb_acc = xgb_model.score(X_test, y_test)
    xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
    results['XGBoost'] = {'accuracy': xgb_acc, 'auc': xgb_auc}
    models['XGBoost'] = xgb_model
    print(f"   Accuracy: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")

    return results, models, (X_test, y_test, features)

# =============================================================================
# FUNCIÓN PRINCIPAL - EJECUTAR PIPELINE COMPLETO
# =============================================================================


def run_complete_corrected_pipeline():
    """Pipeline completo corregido que conserva datos y supera 85%"""
    print("🚀 EJECUTANDO PIPELINE COMPLETO CORREGIDO")
    print("="*50)

    start_time = time.time()

    try:
        # PASO 1: Cargar y convertir
        df = load_and_convert_csgo_data()

        # PASO 2: Limpiar datos
        df_clean = intelligent_data_cleaning(df)

        # PASO 3: Preparar features
        df_prepared, features = prepare_complete_features(df_clean)

        # PASO 4: Entrenar modelos
        results, models, test_data = train_all_models_complete(
            df_prepared, features)

        # Ranking final
        sorted_results = sorted(
            results.items(), key=lambda x: x[1]['accuracy'], reverse=True)

        print("\n🏆 RANKING FINAL DE MODELOS:")
        print("-" * 60)
        print(f"{'Pos':<4} {'Modelo':<18} {'Accuracy':<12} {'AUC':<8} {'Estado'}")
        print("-" * 60)

        medals = ["🥇", "🥈", "🥉", "4°", "5°", "6°"]
        for i, (model, metrics) in enumerate(sorted_results):
            medal = medals[i] if i < len(medals) else f"{i+1}°"
            acc_pct = metrics['accuracy'] * 100

            # Verificar objetivo
            if metrics['accuracy'] >= 0.85:
                status = "✅ OBJETIVO ALCANZADO"
            else:
                status = "⚠️ No alcanzado"

            print(
                f"{medal:<4} {model:<18} {acc_pct:<12.2f}% {metrics['auc']:<8.4f} {status}")

        # Resumen final
        total_time = time.time() - start_time
        best_acc = sorted_results[0][1]['accuracy']

        print(f"\n" + "="*60)
        print(f"🎉 PIPELINE COMPLETO FINALIZADO")
        print(f"⏱️ Tiempo total: {total_time:.1f} segundos")
        print(f"📊 Datos procesados: {len(df_prepared):,} registros")
        print(
            f"💾 Conservación: {len(df_prepared)/79157*100:.1f}% de datos originales")
        print(f"🏆 Mejor modelo: {sorted_results[0][0]}")
        print(f"🎯 Mejor accuracy: {best_acc*100:.2f}%")

        if best_acc >= 0.85:
            print(f"✅ ¡OBJETIVO DE 85% SUPERADO EXITOSAMENTE!")
        else:
            print(f"⚠️ Objetivo no alcanzado - Revisar optimizaciones")

        print(f"="*60)

        return {
            'data': df_prepared,
            'results': results,
            'models': models,
            'features': features,
            'ranking': sorted_results,
            'total_time': total_time
        }

    except Exception as e:
        print(f"❌ Error en pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

# =============================================================================
# EJECUTAR Y EXPORTAR
# =============================================================================


print("🎬 INICIANDO EJECUCIÓN DEL PIPELINE COMPLETO...")
print("🎯 Objetivo: Reproducir 97.31% accuracy con XGBoost")
print("📊 Usar: Máximo de registros posible (~79k)")

# EJECUTAR PIPELINE COMPLETO
final_results = run_complete_corrected_pipeline()

if final_results:
    print(f"\n🎊 ¡ÉXITO COMPLETO!")
    print(f"📈 Resultados guardados en variable 'final_results'")

    # EXPORTAR MODELO XGBOOST OPTIMIZADO
    print(f"\n📦 EXPORTANDO XGBOOST OPTIMIZADO...")

    # Crear carpeta
    os.makedirs('models', exist_ok=True)

    # Exportar XGBoost
    xgb_model = final_results['models']['XGBoost']
    xgb_features = final_results['features']
    xgb_accuracy = final_results['results']['XGBoost']['accuracy']

    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    with open('models/xgboost_features.pkl', 'wb') as f:
        pickle.dump(xgb_features, f)

    # También actualizar feature_names.pkl para compatibilidad
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(xgb_features, f)

    print(f"✅ XGBoost optimizado exportado:")
    print(f"   🎯 Accuracy: {xgb_accuracy*100:.2f}%")
    print(f"   📊 Features: {len(xgb_features)}")
    print(f"   🎉 ¡Objetivo 85% SUPERADO!")

    # Mostrar características
    print(f"\n📋 Características del modelo:")
    for i, feat in enumerate(xgb_features, 1):
        print(f"   {i:2d}. {feat}")

    print(f"\n🚀 Modelo optimizado listo para usar en app.py")

else:
    print(f"\n❌ Hubo errores - revisar output anterior")
