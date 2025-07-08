# =============================================================================
# ENTRENAMIENTO FINAL CORREGIDO - SIN ERRORES DE ENCODING
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import xgboost as xgb
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


def convert_timealive_correct(time_str):
    """Conversión correcta: microsegundos a segundos"""
    try:
        if pd.isna(time_str):
            return np.nan
        val_numeric = int(str(time_str).replace('.', ''))
        return val_numeric / 1_000_000
    except:
        return np.nan


def safe_label_encode(series, column_name):
    """Codificación segura que maneja tipos mixtos"""
    try:
        # Convertir todo a string primero
        series_str = series.astype(str).fillna('Missing')

        # Crear encoder
        le = LabelEncoder()
        encoded = le.fit_transform(series_str)

        print(f"   ✅ {column_name}: {len(le.classes_)} categorías únicas")
        return encoded, le

    except Exception as e:
        print(
            f"   ⚠️ {column_name}: Error en encoding, usando valores por defecto")
        return np.zeros(len(series)), None


def create_robust_features(df):
    """Feature engineering robusto sin errores"""
    print("🔧 FEATURE ENGINEERING ROBUSTO")
    print("="*40)

    df_enhanced = df.copy()

    # 1. CONVERSIÓN DE TimeAlive
    print("⏱️ Convirtiendo TimeAlive...")
    df_enhanced['TimeAlive_seconds'] = df_enhanced['TimeAlive'].apply(
        convert_timealive_correct)

    valid_times = df_enhanced['TimeAlive_seconds'].dropna()
    print(f"   Valores válidos: {len(valid_times):,}")

    # 2. FILTRADO REALISTA
    print("🧹 Filtrando valores realistas...")
    initial_count = len(df_enhanced)

    # Filtro basado en análisis previo: microsegundos funcionan mejor
    df_enhanced = df_enhanced[
        (df_enhanced['TimeAlive_seconds'].notna()) &
        (df_enhanced['TimeAlive_seconds'] > 0) &
        (df_enhanced['TimeAlive_seconds'] <= 300)  # Máximo 5 minutos
    ]

    final_count = len(df_enhanced)
    print(
        f"   Registros: {final_count:,} de {initial_count:,} ({final_count/initial_count*100:.1f}%)")

    # Si muy pocos datos, usar filtro más permisivo
    if final_count < 1000:
        print("   🔧 Aplicando filtro más permisivo...")
        df_enhanced = df[
            (df['TimeAlive_seconds'].notna()) &
            (df['TimeAlive_seconds'] > 0) &
            (df['TimeAlive_seconds'] <= 1000)
        ]
        final_count = len(df_enhanced)
        print(f"   Registros con filtro permisivo: {final_count:,}")

    # 3. CODIFICACIÓN SEGURA DE CATEGÓRICAS
    print("🏷️ Codificando variables categóricas...")

    # Mapas
    if 'Map' in df_enhanced.columns:
        df_enhanced['Map_Encoded'], _ = safe_label_encode(
            df_enhanced['Map'], 'Map')

    # Teams
    if 'Team' in df_enhanced.columns:
        df_enhanced['Team_Encoded'], _ = safe_label_encode(
            df_enhanced['Team'], 'Team')

    # Winners (con manejo robusto de tipos mixtos)
    if 'RoundWinner' in df_enhanced.columns:
        df_enhanced['RoundWinner_Encoded'], _ = safe_label_encode(
            df_enhanced['RoundWinner'], 'RoundWinner')

    if 'MatchWinner' in df_enhanced.columns:
        df_enhanced['MatchWinner_Encoded'], _ = safe_label_encode(
            df_enhanced['MatchWinner'], 'MatchWinner')

    # 4. CARACTERÍSTICAS DE RENDIMIENTO
    print("📊 Creando características de rendimiento...")

    # Ratios básicos
    df_enhanced['HeadshotRatio'] = np.where(
        df_enhanced['MatchKills'] > 0,
        df_enhanced['MatchHeadshots'] / df_enhanced['MatchKills'],
        0
    )

    df_enhanced['RoundHeadshotRatio'] = np.where(
        df_enhanced['RoundKills'] > 0,
        df_enhanced['RoundHeadshots'] / df_enhanced['RoundKills'],
        0
    )

    # Assists (si existe)
    if 'MatchAssists' in df_enhanced.columns:
        df_enhanced['AssistRatio'] = np.where(
            (df_enhanced['MatchKills'] + df_enhanced['MatchAssists']) > 0,
            df_enhanced['MatchAssists'] /
            (df_enhanced['MatchKills'] + df_enhanced['MatchAssists']),
            0
        )

    # 5. CARACTERÍSTICAS DE EQUIPAMIENTO
    print("💰 Características de equipamiento...")

    df_enhanced['EquipmentROI'] = np.where(
        df_enhanced['RoundStartingEquipmentValue'] > 0,
        (df_enhanced['RoundKills'] * 1000) /
        df_enhanced['RoundStartingEquipmentValue'],
        0
    )

    df_enhanced['PersonalEquipmentAdvantage'] = (
        df_enhanced['RoundStartingEquipmentValue'] -
        (df_enhanced['TeamStartingEquipmentValue'] / 5)
    )

    # Categorías de equipamiento
    def equipment_category(value):
        if pd.isna(value) or value < 500:
            return 0
        elif value < 2000:
            return 1
        elif value < 4000:
            return 2
        else:
            return 3

    df_enhanced['EquipmentCategory'] = df_enhanced['RoundStartingEquipmentValue'].apply(
        equipment_category)

    # 6. CARACTERÍSTICAS TÁCTICAS
    print("🎯 Características tácticas...")

    # Agresividad
    df_enhanced['Aggressiveness'] = (
        df_enhanced['RoundKills'] * 3 +
        df_enhanced.get('RoundFlankKills', 0) * 5 +
        df_enhanced['RoundHeadshots'] * 2
    )

    # Eficiencia temporal
    df_enhanced['KillsPerSecond'] = np.where(
        df_enhanced['TimeAlive_seconds'] > 0,
        df_enhanced['RoundKills'] / df_enhanced['TimeAlive_seconds'],
        0
    )

    df_enhanced['KillsPerRound'] = df_enhanced['MatchKills'] / \
        (df_enhanced['RoundId'] + 1)

    # 7. INTERACCIONES
    print("🔗 Interacciones...")

    df_enhanced['KillsEquipmentInteraction'] = (
        df_enhanced['MatchKills'] * df_enhanced['EquipmentROI']
    )

    df_enhanced['HeadshotEquipmentInteraction'] = (
        df_enhanced['HeadshotRatio'] * df_enhanced['EquipmentCategory']
    )

    df_enhanced['TimePerformanceInteraction'] = (
        df_enhanced['TimeAlive_seconds'] * df_enhanced['KillsPerSecond']
    )

    # 8. LIMPIEZA FINAL
    print("🧹 Limpieza final...")

    # Reemplazar infinitos y NaN
    numeric_columns = df_enhanced.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df_enhanced[col] = df_enhanced[col].replace([np.inf, -np.inf], 0)
        df_enhanced[col] = df_enhanced[col].fillna(0)

    print(f"✅ Features creadas: {len(df_enhanced.columns)} columnas")

    return df_enhanced


def train_optimized_models():
    """Entrenamiento optimizado sin errores"""
    print("🎮 ENTRENAMIENTO OPTIMIZADO - SIN ERRORES")
    print("="*55)

    # 1. CARGAR DATOS
    print("📁 Cargando datos...")
    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
    print(f"✅ Datos cargados: {df.shape[0]:,} registros")

    # 2. FEATURE ENGINEERING
    df_enhanced = create_robust_features(df)

    # 3. PREPARAR VARIABLES OBJETIVO
    print(f"\n🎯 Preparando variables objetivo...")

    # Regresión: TimeAlive en segundos
    y_regression = df_enhanced['TimeAlive_seconds']

    # Clasificación: Survived (más confiable)
    if 'Survived' in df_enhanced.columns:
        # Convertir Survived a numérico de forma segura
        y_classification = df_enhanced['Survived'].astype(str)
        y_classification = (y_classification == 'True').astype(int)
    else:
        # Fallback: usar mediana
        median_time = y_regression.median()
        y_classification = (y_regression > median_time).astype(int)

    print(
        f"   Regresión - rango: {y_regression.min():.2f} - {y_regression.max():.2f} segundos")
    print(
        f"   Clasificación - distribución: {y_classification.value_counts().to_dict()}")

    # 4. SELECCIONAR CARACTERÍSTICAS
    print(f"\n🔧 Seleccionando características...")

    # Excluir columnas problemáticas
    exclude_cols = [
        'TimeAlive', 'TimeAlive_seconds', 'Survived', 'MatchId', 'RoundId',
        'Unnamed: 0', 'Team', 'Map', 'RoundWinner', 'MatchWinner', 'AbnormalMatch',
        'InternalTeamId', 'FirstKillTime', 'TravelledDistance', 'PrimaryWeapon'
    ]

    feature_cols = [col for col in df_enhanced.columns
                    if col not in exclude_cols and df_enhanced[col].dtype in ['int64', 'float64']]

    X = df_enhanced[feature_cols].copy()
    print(f"   Características seleccionadas: {len(feature_cols)}")

    # Verificar que no hay NaN o infinitos
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # 5. ENTRENAR RANDOM FOREST (REGRESIÓN)
    print(f"\n🌳 ENTRENANDO RANDOM FOREST (REGRESIÓN)...")

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )

    # Modelo optimizado pero simple
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_reg_train)
    y_reg_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_reg_test, y_reg_pred)

    print(f"   ✅ Random Forest R²: {rf_r2:.3f}")

    # 6. ENTRENAR XGBOOST (CLASIFICACIÓN)
    print(f"\n🚀 ENTRENANDO XGBOOST (CLASIFICACIÓN)...")

    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )

    # Modelo optimizado
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )

    xgb_model.fit(X_train, y_class_train)
    y_class_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_class_test, y_class_proba)

    print(f"   ✅ XGBoost AUC: {xgb_auc:.3f}")

    # 7. GUARDAR MODELOS
    print(f"\n💾 Guardando modelos...")
    os.makedirs('backend/models', exist_ok=True)

    with open('backend/models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    with open('backend/models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    with open('backend/models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    print("✅ Modelos guardados")

    # 8. MOSTRAR IMPORTANCIA DE CARACTERÍSTICAS
    print(f"\n🎯 TOP 10 CARACTERÍSTICAS MÁS IMPORTANTES:")
    print("Random Forest:")
    rf_importance = list(zip(feature_cols, rf_model.feature_importances_))
    rf_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(rf_importance[:5], 1):
        print(f"   {i}. {feature}: {importance:.3f}")

    print("XGBoost:")
    xgb_importance = list(zip(feature_cols, xgb_model.feature_importances_))
    xgb_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(xgb_importance[:5], 1):
        print(f"   {i}. {feature}: {importance:.3f}")

    # 9. EVALUACIÓN FINAL
    print(f"\n🎉 RESULTADOS FINALES")
    print("="*50)
    print(f"📊 Registros utilizados: {len(df_enhanced):,}")
    print(f"🔧 Características: {len(feature_cols)}")
    print(f"🌳 Random Forest R²: {rf_r2:.3f}")
    print(f"🚀 XGBoost AUC: {xgb_auc:.3f}")

    # Verificar objetivos
    rf_target = rf_r2 >= 0.70
    xgb_target = xgb_auc >= 0.85

    print(f"\n🎯 EVALUACIÓN DE OBJETIVOS:")
    print(
        f"   Random Forest R² ≥ 70%: {'✅ ALCANZADO' if rf_target else '❌ NO ALCANZADO'} ({rf_r2:.1%})")
    print(
        f"   XGBoost AUC ≥ 85%: {'✅ ALCANZADO' if xgb_target else '❌ NO ALCANZADO'} ({xgb_auc:.1%})")

    if rf_target and xgb_target:
        print(f"\n🏆 ¡AMBOS OBJETIVOS ALCANZADOS!")
    elif rf_target or xgb_target:
        print(f"\n🟡 Un objetivo alcanzado - Mejora significativa")
    else:
        print(f"\n🔴 Objetivos no alcanzados pero modelos funcionales")

    print(f"\n🌐 PRÓXIMOS PASOS:")
    print(f"   cd backend")
    print(f"   python app.py")
    print(f"   Abrir http://localhost:5000")

    return {
        'rf_r2': rf_r2,
        'xgb_auc': xgb_auc,
        'records_used': len(df_enhanced),
        'features_used': len(feature_cols)
    }


if __name__ == "__main__":
    results = train_optimized_models()

    print(f"\n✅ Entrenamiento completado exitosamente")
    print(f"📈 Modelos mejorados y guardados para la aplicación web")
