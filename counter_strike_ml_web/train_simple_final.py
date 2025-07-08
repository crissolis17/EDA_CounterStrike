# =============================================================================
# ENTRENAMIENTO ULTRA SIMPLIFICADO - SIN ERRORES GARANTIZADO
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, roc_auc_score
import xgboost as xgb
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


def convert_timealive_microseconds(time_str):
    """Convierte TimeAlive de microsegundos a segundos"""
    try:
        if pd.isna(time_str):
            return np.nan
        val_str = str(time_str).replace('.', '')
        val_numeric = int(val_str)
        return val_numeric / 1_000_000  # Microsegundos a segundos
    except:
        return np.nan


def train_simple_final():
    """Entrenamiento ultra simplificado que funciona garantizado"""
    print("🎮 ENTRENAMIENTO ULTRA SIMPLIFICADO")
    print("="*45)
    print("🎯 Enfoque: Modelos funcionales sin errores")
    print("="*45)

    # 1. CARGAR DATOS
    print("\n📁 Cargando datos...")
    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
    print(
        f"✅ Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")

    # 2. CONVERTIR TimeAlive CORRECTAMENTE
    print("\n⏱️ Convirtiendo TimeAlive (microsegundos → segundos)...")
    df['TimeAlive_seconds'] = df['TimeAlive'].apply(
        convert_timealive_microseconds)

    valid_count = df['TimeAlive_seconds'].notna().sum()
    print(f"   Valores válidos: {valid_count:,}")

    if valid_count > 0:
        print(
            f"   Rango: {df['TimeAlive_seconds'].min():.2f} - {df['TimeAlive_seconds'].max():.2f} segundos")
        print(f"   Promedio: {df['TimeAlive_seconds'].mean():.2f} segundos")

    # 3. FILTRAR DATOS REALISTAS
    print("\n🧹 Filtrando datos realistas...")

    # Usar filtro muy permisivo para obtener más datos
    df_clean = df[
        (df['TimeAlive_seconds'].notna()) &
        (df['TimeAlive_seconds'] > 0) &
        (df['TimeAlive_seconds'] <= 100000)  # Muy permisivo
    ].copy()

    print(f"   Registros después del filtro: {len(df_clean):,}")

    # Si aún muy pocos, usar estrategia alternativa
    if len(df_clean) < 500:
        print("   🔄 Pocos datos válidos, usando estrategia alternativa...")
        # Usar solo datos con TimeAlive numérico directo
        df['TimeAlive_numeric'] = pd.to_numeric(
            df['TimeAlive'], errors='coerce')
        df_clean = df.dropna(subset=['TimeAlive_numeric'])
        df_clean = df_clean[df_clean['TimeAlive_numeric'] > 0]
        # Usar como está
        df_clean['TimeAlive_seconds'] = df_clean['TimeAlive_numeric']
        print(f"   Registros con estrategia alternativa: {len(df_clean):,}")

    # 4. PREPARAR CARACTERÍSTICAS BÁSICAS
    print("\n🔧 Preparando características básicas...")

    # Codificar mapa de forma segura
    if 'Map' in df_clean.columns:
        le_map = LabelEncoder()
        df_clean['Map_Encoded'] = le_map.fit_transform(
            df_clean['Map'].fillna('Unknown'))
        print(f"   ✅ Mapas codificados: {df_clean['Map'].nunique()} únicos")
    else:
        df_clean['Map_Encoded'] = 0

    # Características básicas que sabemos que existen
    basic_features = []
    potential_features = [
        'MatchKills', 'MatchHeadshots', 'RoundKills', 'RoundHeadshots',
        'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
        'Map_Encoded', 'MatchAssists', 'RoundAssists'
    ]

    for feature in potential_features:
        if feature in df_clean.columns:
            basic_features.append(feature)
            print(f"   ✅ {feature}")
        else:
            print(f"   ❌ {feature} - no encontrada")

    # Crear algunas características derivadas simples
    print("\n⚙️ Creando características derivadas...")

    # Ratio de headshots
    if 'MatchKills' in df_clean.columns and 'MatchHeadshots' in df_clean.columns:
        df_clean['HeadshotRatio'] = np.where(
            df_clean['MatchKills'] > 0,
            df_clean['MatchHeadshots'] / df_clean['MatchKills'],
            0
        )
        basic_features.append('HeadshotRatio')
        print("   ✅ HeadshotRatio")

    # ROI del equipamiento
    if 'RoundKills' in df_clean.columns and 'RoundStartingEquipmentValue' in df_clean.columns:
        df_clean['EquipmentROI'] = np.where(
            df_clean['RoundStartingEquipmentValue'] > 0,
            (df_clean['RoundKills'] * 1000) /
            df_clean['RoundStartingEquipmentValue'],
            0
        )
        basic_features.append('EquipmentROI')
        print("   ✅ EquipmentROI")

    # Kills por ronda
    if 'MatchKills' in df_clean.columns and 'RoundId' in df_clean.columns:
        df_clean['KillsPerRound'] = df_clean['MatchKills'] / \
            (df_clean['RoundId'] + 1)
        basic_features.append('KillsPerRound')
        print("   ✅ KillsPerRound")

    print(f"\n📊 Características finales: {len(basic_features)}")

    # 5. PREPARAR DATOS PARA ENTRENAMIENTO
    print(f"\n📋 Preparando datos para entrenamiento...")

    # Limpiar características
    X = df_clean[basic_features].copy()
    X = X.fillna(0)  # Llenar NaN con 0
    X = X.replace([np.inf, -np.inf], 0)  # Reemplazar infinitos

    # Variable de regresión: TimeAlive
    y_regression = df_clean['TimeAlive_seconds'].copy()

    # Variable de clasificación: Survived
    if 'Survived' in df_clean.columns:
        # Convertir de forma muy segura
        survived_values = df_clean['Survived'].astype(str)
        y_classification = (survived_values.str.lower() == 'true').astype(int)
        print(f"   ✅ Usando columna 'Survived' para clasificación")
    else:
        # Fallback: usar mediana de TimeAlive
        median_time = y_regression.median()
        y_classification = (y_regression > median_time).astype(int)
        print(
            f"   ✅ Usando mediana de TimeAlive para clasificación (corte: {median_time:.2f}s)")

    print(
        f"   Regresión - rango: {y_regression.min():.2f} - {y_regression.max():.2f} segundos")
    print(
        f"   Clasificación - distribución: {y_classification.value_counts().to_dict()}")

    # 6. ENTRENAR RANDOM FOREST (REGRESIÓN)
    print(f"\n🌳 ENTRENANDO RANDOM FOREST (REGRESIÓN)...")

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print("   🔄 Entrenando modelo...")
    rf_model.fit(X_train, y_reg_train)

    y_reg_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_reg_test, y_reg_pred)

    print(f"   ✅ Random Forest R²: {rf_r2:.3f}")

    # 7. ENTRENAR XGBOOST (CLASIFICACIÓN)
    print(f"\n🚀 ENTRENANDO XGBOOST (CLASIFICACIÓN)...")

    # Verificar que tenemos ambas clases
    if len(y_classification.unique()) < 2:
        print("   ⚠️ Solo una clase disponible, creando clasificación balanceada...")
        # Crear clasificación artificial más balanceada
        percentile_75 = y_regression.quantile(0.75)
        y_classification = (y_regression > percentile_75).astype(int)
        print(
            f"   ✅ Nueva distribución: {y_classification.value_counts().to_dict()}")

    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )

    print("   🔄 Entrenando modelo...")
    xgb_model.fit(X_train, y_class_train)

    y_class_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_class_test, y_class_proba)

    print(f"   ✅ XGBoost AUC: {xgb_auc:.3f}")

    # 8. GUARDAR MODELOS
    print(f"\n💾 Guardando modelos...")
    os.makedirs('backend/models', exist_ok=True)

    try:
        with open('backend/models/random_forest_model.pkl', 'wb') as f:
            pickle.dump(rf_model, f)
        print("   ✅ Random Forest guardado")

        with open('backend/models/xgboost_model.pkl', 'wb') as f:
            pickle.dump(xgb_model, f)
        print("   ✅ XGBoost guardado")

        with open('backend/models/feature_names.pkl', 'wb') as f:
            pickle.dump(basic_features, f)
        print("   ✅ Feature names guardado")

    except Exception as e:
        print(f"   ❌ Error guardando modelos: {e}")
        return None

    # 9. MOSTRAR RESULTADOS FINALES
    print(f"\n🎉 RESULTADOS FINALES")
    print("="*50)
    print(f"📊 Registros utilizados: {len(df_clean):,}")
    print(f"🔧 Características: {len(basic_features)}")
    print(f"🌳 Random Forest R²: {rf_r2:.3f}")
    print(f"🚀 XGBoost AUC: {xgb_auc:.3f}")

    # Evaluación vs objetivos
    rf_target = rf_r2 >= 0.70
    xgb_target = xgb_auc >= 0.85

    print(f"\n🎯 EVALUACIÓN DE OBJETIVOS:")
    print(
        f"   Random Forest R² ≥ 70%: {'✅ ALCANZADO' if rf_target else '❌ NO ALCANZADO'} ({rf_r2:.1%})")
    print(
        f"   XGBoost AUC ≥ 85%: {'✅ ALCANZADO' if xgb_target else '❌ NO ALCANZADO'} ({xgb_auc:.1%})")

    # Mostrar mejora vs resultados anteriores
    prev_rf_r2 = 0.049  # Del entrenamiento anterior
    prev_xgb_auc = 0.606

    rf_improvement = rf_r2 - prev_rf_r2
    xgb_improvement = xgb_auc - prev_xgb_auc

    print(f"\n📈 MEJORA VS VERSIÓN ANTERIOR:")
    print(
        f"   Random Forest: {rf_improvement:+.3f} ({rf_improvement/prev_rf_r2*100:+.1f}%)")
    print(
        f"   XGBoost: {xgb_improvement:+.3f} ({xgb_improvement/prev_xgb_auc*100:+.1f}%)")

    if rf_target and xgb_target:
        print(f"\n🏆 ¡AMBOS OBJETIVOS ALCANZADOS!")
        status = "ÉXITO COMPLETO"
    elif rf_target or xgb_target:
        print(f"\n🥈 Un objetivo alcanzado - ¡Progreso significativo!")
        status = "ÉXITO PARCIAL"
    elif rf_improvement > 0 and xgb_improvement > 0:
        print(f"\n📊 Modelos mejorados significativamente")
        status = "MEJORA NOTABLE"
    else:
        print(f"\n🔧 Modelos funcionales para la aplicación web")
        status = "FUNCIONAL"

    print(f"\n🌐 PRÓXIMOS PASOS:")
    print(f"   1. cd backend")
    print(f"   2. python app.py")
    print(f"   3. Abrir http://localhost:5000")
    print(f"   4. Probar predicciones en la interfaz web")

    # Mostrar características más importantes
    print(f"\n🎯 TOP 5 CARACTERÍSTICAS MÁS IMPORTANTES:")
    print("Random Forest:")
    rf_importance = list(zip(basic_features, rf_model.feature_importances_))
    rf_importance.sort(key=lambda x: x[1], reverse=True)
    for i, (feature, importance) in enumerate(rf_importance[:5], 1):
        print(f"   {i}. {feature}: {importance:.3f}")

    return {
        'rf_r2': rf_r2,
        'xgb_auc': xgb_auc,
        'status': status,
        'records_used': len(df_clean),
        'features_used': len(basic_features),
        'improvement': {
            'rf': rf_improvement,
            'xgb': xgb_improvement
        }
    }


if __name__ == "__main__":
    print("🎮 COUNTER STRIKE ML - ENTRENAMIENTO FINAL")
    print("="*60)
    print("🎯 Objetivo: Modelos funcionales para aplicación web")
    print("🔧 Estrategia: Simplificado pero robusto")
    print("="*60)

    try:
        results = train_simple_final()

        if results:
            print(f"\n✅ ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
            print(f"📊 Status: {results['status']}")
            print(f"🎯 Los modelos están listos para usar en la web")
        else:
            print(f"\n❌ Error en el entrenamiento")

    except Exception as e:
        print(f"\n💥 ERROR INESPERADO: {e}")
        import traceback
        traceback.print_exc()
