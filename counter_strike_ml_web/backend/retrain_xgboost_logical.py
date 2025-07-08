#!/usr/bin/env python3
# =============================================================================
# REENTRENAR XGBOOST CON LÓGICA CORRECTA
# =============================================================================

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    print("✅ XGBoost disponible")
except ImportError:
    print("❌ XGBoost no disponible")
    exit(1)

print("🚀 REENTRENANDO XGBOOST CON LÓGICA CORRECTA")
print("="*50)
print("🎯 Objetivo: Equipamiento alto + rifle = Alta supervivencia")
print("📊 Dataset: Tu archivo original")
print("="*50)

# =============================================================================
# CARGAR Y PREPARAR DATOS CON LÓGICA CORRECTA
# =============================================================================


def load_and_prepare_data_with_correct_logic():
    """Cargar datos y crear variable objetivo con lógica correcta"""
    print("\n📂 CARGANDO DATOS...")

    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
    print(f"✅ Datos cargados: {df.shape[0]:,} registros")

    # Limpiar equipamiento
    def clean_equipment(value):
        if pd.isna(value):
            return 0
        try:
            cleaned = str(value).replace('.', '').replace(',', '.')
            return float(cleaned)
        except:
            return 0

    df['RoundStartingEquipmentValue'] = df['RoundStartingEquipmentValue'].apply(
        clean_equipment)
    df['TeamStartingEquipmentValue'] = df['TeamStartingEquipmentValue'].apply(
        clean_equipment)

    # Limpiar variables categóricas y numéricas
    numeric_cols = ['MatchKills', 'RoundKills', 'MatchAssists', 'RoundAssists',
                    'MatchHeadshots', 'RoundHeadshots', 'MatchFlankKills', 'RoundFlankKills',
                    'TravelledDistance', 'FirstKillTime', 'RLethalGrenadesThrown',
                    'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle',
                    'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_equipment)

    print(f"✅ Datos limpiados")

    return df


def create_logical_survival_target(df):
    """Crear variable objetivo basada en LÓGICA CORRECTA de CS:GO"""
    print("\n🎯 CREANDO VARIABLE OBJETIVO CON LÓGICA CORRECTA...")

    # Calcular score de supervivencia basado en equipamiento y arma
    survival_scores = []

    for _, row in df.iterrows():
        score = 0.0

        # Factor 1: Equipamiento personal (40% peso)
        equipment = row.get('RoundStartingEquipmentValue', 0)
        if equipment >= 12000:
            score += 0.4
        elif equipment >= 8000:
            score += 0.3
        elif equipment >= 4000:
            score += 0.2
        elif equipment >= 2000:
            score += 0.1

        # Factor 2: Equipamiento del equipo (30% peso)
        team_equipment = row.get('TeamStartingEquipmentValue', 0)
        if team_equipment >= 60000:
            score += 0.3
        elif team_equipment >= 40000:
            score += 0.25
        elif team_equipment >= 25000:
            score += 0.15
        elif team_equipment >= 15000:
            score += 0.1

        # Factor 3: Tipo de arma (30% peso)
        rifle = row.get('PrimaryAssaultRifle', 0)
        sniper = row.get('PrimarySniperRifle', 0)
        smg = row.get('PrimarySMG', 0)
        pistol = row.get('PrimaryPistol', 0)

        if rifle > 0.5:
            score += 0.3
        elif sniper > 0.5:
            score += 0.25
        elif smg > 0.5:
            score += 0.15
        elif pistol > 0.5:
            score += 0.05

        survival_scores.append(score)

    # Convertir scores a clasificación binaria
    survival_scores = np.array(survival_scores)

    # Umbral: score >= 0.5 = Alta supervivencia
    df['SurvivalClass_Logical'] = (survival_scores >= 0.5).astype(int)

    # Estadísticas
    class_counts = df['SurvivalClass_Logical'].value_counts()
    print(f"📊 Distribución de clases lógicas:")
    print(
        f"   Baja supervivencia (0): {class_counts.get(0, 0):,} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(
        f"   Alta supervivencia (1): {class_counts.get(1, 0):,} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")

    return df


def prepare_features_for_xgboost(df):
    """Preparar características para XGBoost"""
    print("\n⚙️ PREPARANDO CARACTERÍSTICAS...")

    # Features numéricas
    numeric_features = []
    numeric_candidates = [
        'MatchKills', 'RoundKills', 'MatchAssists', 'RoundAssists',
        'MatchHeadshots', 'RoundHeadshots', 'MatchFlankKills', 'RoundFlankKills',
        'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
        'TravelledDistance', 'FirstKillTime', 'RLethalGrenadesThrown',
        'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle',
        'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol'
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            if df[col].var() > 0:
                numeric_features.append(col)
                print(f"   ✅ {col}")

    # Features derivadas
    derived_features = []
    print(f"\n🔧 Creando features derivadas:")

    if 'MatchKills' in numeric_features:
        df['KD_Ratio'] = df['MatchKills'] / (df['MatchKills'] + 1)
        derived_features.append('KD_Ratio')
        print(f"   ✅ KD_Ratio")

    if 'MatchHeadshots' in numeric_features and 'MatchKills' in numeric_features:
        df['Headshot_Efficiency'] = df['MatchHeadshots'] / \
            (df['MatchKills'] + 1)
        derived_features.append('Headshot_Efficiency')
        print(f"   ✅ Headshot_Efficiency")

    if 'RoundStartingEquipmentValue' in numeric_features and 'MatchKills' in numeric_features:
        df['Equipment_ROI'] = df['MatchKills'] / \
            (df['RoundStartingEquipmentValue'] / 1000 + 1)
        derived_features.append('Equipment_ROI')
        print(f"   ✅ Equipment_ROI")

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
        print(f"   ✅ Map_Encoded")

    if 'Team' in df.columns:
        le_team = LabelEncoder()
        df['Team_Encoded'] = le_team.fit_transform(
            df['Team'].fillna('Unknown'))
        categorical_features.append('Team_Encoded')
        print(f"   ✅ Team_Encoded")

    # Limpiar features derivadas
    for feat in derived_features:
        df[feat] = df[feat].replace([np.inf, -np.inf], 0).fillna(0)

    all_features = numeric_features + derived_features + categorical_features

    print(f"\n📊 Total features: {len(all_features)}")
    return df, all_features


def train_logical_xgboost(df, features):
    """Entrenar XGBoost con lógica correcta"""
    print("\n🚀 ENTRENANDO XGBOOST CON LÓGICA CORRECTA...")

    # Preparar datos
    X = df[features].fillna(0)
    y = df['SurvivalClass_Logical']

    print(f"📊 Datos para entrenamiento:")
    print(f"   Características: {X.shape[1]}")
    print(f"   Registros: {X.shape[0]:,}")
    print(f"   Balance: {y.value_counts().to_dict()}")

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calcular balance de clases
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Parámetros XGBoost optimizados
    xgb_params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'eval_metric': 'auc',
        'verbosity': 0
    }

    # Entrenar modelo
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)

    # Evaluar
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n📊 RESULTADOS:")
    print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   AUC: {auc:.4f}")

    return xgb_model, accuracy, auc, features

# =============================================================================
# EJECUTAR PIPELINE COMPLETO
# =============================================================================


print("🚀 EJECUTANDO REENTRENAMIENTO...")

try:
    # Cargar y preparar datos
    df = load_and_prepare_data_with_correct_logic()
    df = create_logical_survival_target(df)
    df, features = prepare_features_for_xgboost(df)

    # Entrenar XGBoost con lógica correcta
    xgb_model, accuracy, auc, xgb_features = train_logical_xgboost(
        df, features)

    # Exportar modelo lógico
    print(f"\n📦 EXPORTANDO XGBOOST CON LÓGICA CORRECTA...")

    os.makedirs('models', exist_ok=True)

    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    with open('models/xgboost_features.pkl', 'wb') as f:
        pickle.dump(xgb_features, f)

    print(f"✅ XGBoost lógico exportado:")
    print(f"   🎯 Accuracy: {accuracy*100:.2f}%")
    print(f"   📊 AUC: {auc:.4f}")
    print(f"   📋 Features: {len(xgb_features)}")

    # Prueba rápida
    print(f"\n🧪 PRUEBA RÁPIDA:")

    # Configuración alta
    X_test_high = np.zeros((1, len(xgb_features)))
    if 'RoundStartingEquipmentValue' in xgb_features:
        idx = xgb_features.index('RoundStartingEquipmentValue')
        X_test_high[0, idx] = 16000
    if 'TeamStartingEquipmentValue' in xgb_features:
        idx = xgb_features.index('TeamStartingEquipmentValue')
        X_test_high[0, idx] = 80000
    if 'PrimaryAssaultRifle' in xgb_features:
        idx = xgb_features.index('PrimaryAssaultRifle')
        X_test_high[0, idx] = 1.0

    pred_high = xgb_model.predict(X_test_high)[0]
    proba_high = xgb_model.predict_proba(X_test_high)[0]

    # Configuración baja
    X_test_low = np.zeros((1, len(xgb_features)))
    if 'RoundStartingEquipmentValue' in xgb_features:
        idx = xgb_features.index('RoundStartingEquipmentValue')
        X_test_low[0, idx] = 800
    if 'TeamStartingEquipmentValue' in xgb_features:
        idx = xgb_features.index('TeamStartingEquipmentValue')
        X_test_low[0, idx] = 10000
    if 'PrimaryPistol' in xgb_features:
        idx = xgb_features.index('PrimaryPistol')
        X_test_low[0, idx] = 1.0

    pred_low = xgb_model.predict(X_test_low)[0]
    proba_low = xgb_model.predict_proba(X_test_low)[0]

    print(
        f"   Equipamiento ALTO + Rifle: Clase {pred_high} (Prob alta: {proba_high[1]:.3f})")
    print(
        f"   Equipamiento BAJO + Pistola: Clase {pred_low} (Prob alta: {proba_low[1]:.3f})")

    if pred_high == 1 and pred_low == 0:
        print(f"\n🎉 ¡LÓGICA CORRECTA IMPLEMENTADA!")
    else:
        print(f"\n⚠️ Revisar lógica - puede necesitar ajustes")

    print(f"\n🚀 Reinicia servidor: python app.py")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
