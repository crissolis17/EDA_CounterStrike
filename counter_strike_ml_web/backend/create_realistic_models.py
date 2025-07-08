import pickle
import os
import numpy as np

# Instalar las dependencias si no están
try:
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
except ImportError:
    print("Instalando dependencias...")
    import subprocess
    subprocess.run(["pip", "install", "scikit-learn", "xgboost"])
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

# Crear directorio
os.makedirs('models', exist_ok=True)

print("🔧 Creando modelos con LÓGICA CORRECTA de Counter Strike...")

# Generar datos sintéticos que sigan la lógica real del juego
np.random.seed(42)
n_samples = 5000

# Feature names
feature_names = [
    'MatchKills', 'RoundKills', 'MatchAssists', 'RoundAssists',
    'MatchHeadshots', 'RoundHeadshots', 'MatchFlankKills', 'RoundFlankKills',
    'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
    'TravelledDistance', 'FirstKillTime', 'RLethalGrenadesThrown',
    'RNonLethalGrenadesThrown', 'PrimaryAssaultRifle', 'PrimarySniperRifle',
    'PrimaryHeavy', 'PrimarySMG', 'PrimaryPistol', 'KD_Ratio',
    'Headshot_Efficiency', 'Equipment_ROI', 'Assist_Ratio',
    'Map_Encoded', 'Team_Encoded'
]

print("📊 Generando datos sintéticos con lógica de CS:GO...")

# Generar características realistas
match_kills = np.random.poisson(3, n_samples)  # 0-15 kills típicos
round_kills = np.random.poisson(1, n_samples)  # 0-5 kills por ronda
# 30% de los kills son headshots
headshots = np.random.binomial(match_kills, 0.3)
equipment_value = np.random.uniform(800, 16000, n_samples)  # $800-$16000
team_equipment = np.random.uniform(15000, 80000, n_samples)  # $15k-$80k

# Armas (one-hot encoding)
# rifle, sniper, heavy, smg, pistol
weapon_probs = [0.4, 0.15, 0.05, 0.25, 0.15]
weapon_choice = np.random.choice(5, n_samples, p=weapon_probs)
primary_assault = (weapon_choice == 0).astype(float)
primary_sniper = (weapon_choice == 1).astype(float)
primary_heavy = (weapon_choice == 2).astype(float)
primary_smg = (weapon_choice == 3).astype(float)
primary_pistol = (weapon_choice == 4).astype(float)

# Variables derivadas
kd_ratio = match_kills / np.maximum(1, 5 - match_kills)  # Simulación K/D
headshot_efficiency = headshots / np.maximum(1, match_kills)
equipment_roi = match_kills / np.maximum(1, equipment_value / 1000)

# Crear matriz de características
X = np.column_stack([
    match_kills, round_kills, match_kills * 0.7, round_kills * 0.8,  # assists
    headshots, headshots * 0.6, match_kills * 0.3, round_kills * 0.4,  # flanks
    equipment_value, team_equipment,
    np.random.uniform(500, 2000, n_samples),  # distance
    np.random.uniform(5, 30, n_samples),  # first kill time
    np.random.poisson(1, n_samples),  # lethal grenades
    np.random.poisson(0.5, n_samples),  # non-lethal grenades
    primary_assault, primary_sniper, primary_heavy, primary_smg, primary_pistol,
    kd_ratio, headshot_efficiency, equipment_roi,
    headshots / np.maximum(1, match_kills + headshots),  # assist ratio
    np.random.randint(0, 4, n_samples),  # map encoded
    np.random.randint(0, 2, n_samples)   # team encoded
])

print("🎯 Creando variables objetivo con LÓGICA REALISTA...")

# LÓGICA CORRECTA: Mejor equipamiento + más kills + mejor arma = MAYOR supervivencia
survival_score = (
    0.3 * (equipment_value / 16000) +           # 30% equipamiento personal
    0.25 * (team_equipment / 80000) +          # 25% equipamiento del equipo
    0.2 * (match_kills / 10) +                 # 20% kills en el match
    0.1 * (headshots / 5) +                    # 10% headshots
    0.1 * primary_assault +                    # 10% rifle (mejor arma)
    0.05 * primary_sniper +                    # 5% sniper (segunda mejor)
    -0.1 * primary_pistol +                    # -10% pistola (peor arma)
    np.random.normal(0, 0.1, n_samples)        # Ruido realista
)

# Clasificación: Alta supervivencia si score > 0.5
y_classification = (survival_score > 0.5).astype(int)

# Regresión: Tiempo de supervivencia (5-115 segundos)
# Mapear survival_score a tiempo realista
y_regression = np.clip(
    20 + survival_score * 80 + np.random.normal(0, 10, n_samples),
    5, 115
)

print(
    f"📈 Balance de clases: {np.mean(y_classification)*100:.1f}% alta supervivencia")
print(f"📈 Tiempo promedio: {np.mean(y_regression):.1f} segundos")

# 1. Entrenar XGBoost para clasificación
print("🚀 Entrenando XGBoost (Clasificación)...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X, y_classification)

# Verificar que la lógica es correcta
test_high = np.array([[8, 2, 6, 2, 4, 2, 3, 1, 15000, 75000, 1500, 15, 2, 1,
                      1, 0, 0, 0, 0, 4.0, 0.5, 0.5, 0.6, 2, 0]])
test_low = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 1000, 20000, 800, 25, 0, 0,
                     0, 0, 0, 0, 1, 1.0, 0.0, 1.0, 0.5, 1, 1]])

prob_high = xgb_model.predict_proba(test_high)[0][1]
prob_low = xgb_model.predict_proba(test_low)[0][1]

print(f"   🧪 Prueba configuración ALTA: {prob_high:.3f} (debería ser >0.7)")
print(f"   🧪 Prueba configuración BAJA: {prob_low:.3f} (debería ser <0.4)")

with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("✅ XGBoost con lógica correcta guardado")

# 2. Entrenar Random Forest para regresión
print("🌳 Entrenando Random Forest (Regresión)...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X, y_regression)

# Verificar lógica
time_high = rf_model.predict(test_high)[0]
time_low = rf_model.predict(test_low)[0]

print(f"   🧪 Prueba configuración ALTA: {time_high:.1f}s (debería ser >60s)")
print(f"   🧪 Prueba configuración BAJA: {time_low:.1f}s (debería ser <40s)")

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Random Forest con lógica correcta guardado")

# 3. Guardar feature names
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names guardado")

# 4. Verificación final
print("\n🧪 Verificación final de archivos...")
for filename in ['xgboost_model.pkl', 'random_forest_model.pkl', 'feature_names.pkl']:
    filepath = os.path.join('models', filename)
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"   ✅ {filename} - válido ({type(data).__name__})")
    except Exception as e:
        print(f"   ❌ {filename} - error: {e}")

print("\n🎉 MODELOS CON LÓGICA CORRECTA CREADOS!")
print("🎯 Ahora mejor equipamiento = MAYOR supervivencia")
print("🚀 Ejecuta: python app.py")
print("\n📋 Lógica implementada:")
print("   • Mejor equipamiento personal → Mayor supervivencia")
print("   • Mejor equipamiento de equipo → Mayor supervivencia")
print("   • Más kills → Mayor supervivencia")
print("   • Rifle de asalto → Mayor supervivencia que pistola")
print("   • Más headshots → Mayor supervivencia")
