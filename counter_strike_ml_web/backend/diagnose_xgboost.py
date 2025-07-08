#!/usr/bin/env python3
# =============================================================================
# DIAGNOSTICAR COMPORTAMIENTO DEL MODELO XGBOOST
# =============================================================================

import pickle
import numpy as np

print("🔍 DIAGNOSTICANDO MODELO XGBOOST")
print("="*40)

# Cargar modelo y features
try:
    with open('models/xgboost_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)

    with open('models/xgboost_features.pkl', 'rb') as f:
        features = pickle.load(f)

    print("✅ Modelo y features cargados")
    print(f"📊 Features: {len(features)}")

except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Crear vectores de prueba


def create_test_vector(equipment, team_equipment, weapon_type):
    """Crear vector de prueba"""
    map_encoding = {'de_dust2': 0, 'de_inferno': 1,
                    'de_mirage': 2, 'de_nuke': 3}

    neutral_kills = 2
    neutral_headshots = 1

    features_dict = {
        'MatchKills': float(neutral_kills),
        'RoundKills': 1.0,
        'MatchAssists': 1.0,
        'RoundAssists': 0.0,
        'MatchHeadshots': float(neutral_headshots),
        'RoundHeadshots': 1.0,
        'MatchFlankKills': 1.0,
        'RoundFlankKills': 0.0,
        'RoundStartingEquipmentValue': equipment,
        'TeamStartingEquipmentValue': team_equipment,
        'TravelledDistance': 800.0 + (equipment / 50),
        'FirstKillTime': 15.0,
        'RLethalGrenadesThrown': 1.0 + (equipment > 8000),
        'RNonLethalGrenadesThrown': 0.5 + (equipment > 12000),
        'PrimaryAssaultRifle': 1.0 if weapon_type == 'rifle' else 0.0,
        'PrimarySniperRifle': 1.0 if weapon_type == 'sniper' else 0.0,
        'PrimaryHeavy': 1.0 if weapon_type == 'heavy' else 0.0,
        'PrimarySMG': 1.0 if weapon_type == 'smg' else 0.0,
        'PrimaryPistol': 1.0 if weapon_type == 'pistol' else 0.0,
        'Map_Encoded': 0.0,  # de_dust2
        'Team_Encoded': 0.0,
        'KD_Ratio': 1.5 + (equipment / 10000),
        'Headshot_Efficiency': 0.4 + (0.1 if weapon_type == 'rifle' else 0.0),
        'Equipment_ROI': 2.0 + (equipment / 8000),
        'Assist_Ratio': 0.3,
    }

    # Crear vector ordenado
    feature_vector = []
    for feature_name in features:
        if feature_name in features_dict:
            feature_vector.append(features_dict[feature_name])
        else:
            feature_vector.append(0.0)

    return np.array(feature_vector).reshape(1, -1)


# Tests específicos
tests = [
    ("MÁXIMO", 16000, 80000, "rifle"),
    ("ALTO", 12000, 60000, "rifle"),
    ("MEDIO", 6000, 30000, "smg"),
    ("BAJO", 2000, 15000, "pistol"),
    ("MÍNIMO", 800, 10000, "pistol")
]

print("\n🧪 TESTS DE DIAGNÓSTICO:")
print("-" * 60)
print(f"{'Config':<8} {'Equip':<6} {'Team':<6} {'Arma':<8} {'Clase':<6} {'Prob_0':<8} {'Prob_1':<8}")
print("-" * 60)

for name, equipment, team_equipment, weapon in tests:
    X = create_test_vector(equipment, team_equipment, weapon)

    try:
        # Predicción
        pred_class = xgb_model.predict(X)[0]
        pred_proba = xgb_model.predict_proba(X)[0]

        prob_0 = pred_proba[0]  # Probabilidad clase 0
        prob_1 = pred_proba[1]  # Probabilidad clase 1

        print(f"{name:<8} ${equipment:<5} ${team_equipment:<5} {weapon:<8} {int(pred_class):<6} {prob_0:<8.3f} {prob_1:<8.3f}")

    except Exception as e:
        print(f"{name:<8} ERROR: {e}")

print("\n📊 INTERPRETACIÓN:")
print("   Clase 0 = Baja supervivencia (tiempo < mediana)")
print("   Clase 1 = Alta supervivencia (tiempo > mediana)")
print("\n🎯 ESPERADO:")
print("   Equipamiento MÁXIMO → Clase 1 (alta probabilidad)")
print("   Equipamiento MÍNIMO → Clase 0 (alta probabilidad)")

print(f"\n🔍 SI EL PATRÓN ESTÁ INVERTIDO:")
print(f"   • Equipamiento alto da Clase 0 → Lógica invertida")
print(f"   • Necesitamos invertir las predicciones")
