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

print("🔧 Creando modelos temporales válidos...")

# Datos dummy para entrenar modelos básicos
np.random.seed(42)
X_dummy = np.random.rand(1000, 25)
y_classification = np.random.randint(0, 2, 1000)
y_regression = np.random.rand(1000) * 100

# 1. Crear XGBoost para clasificación
print("📊 Creando XGBoost (Clasificación)...")
xgb_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_dummy, y_classification)

with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("✅ XGBoost guardado")

# 2. Crear Random Forest para regresión
print("📊 Creando Random Forest (Regresión)...")
rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_dummy, y_regression)

with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Random Forest guardado")

# 3. Crear feature names
print("📊 Creando feature names...")
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

with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names guardado")

# 4. Verificar que todos los archivos son válidos
print("\n🧪 Verificando archivos creados...")
for filename in ['xgboost_model.pkl', 'random_forest_model.pkl', 'feature_names.pkl']:
    filepath = os.path.join('models', filename)
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"   ✅ {filename} - válido ({type(data).__name__})")
    except Exception as e:
        print(f"   ❌ {filename} - error: {e}")

print("\n🎉 Modelos temporales creados exitosamente!")
print("🚀 Ahora puedes ejecutar: python app.py")
