# =============================================================================
# ENTRENAMIENTO FINAL CORREGIDO - ANÁLISIS Y CONVERSIÓN CORRECTA
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


def analyze_and_convert_time_alive(df):
    """
    Analiza y convierte TimeAlive al formato correcto
    """
    print("🔍 ANALIZANDO FORMATO DE TimeAlive...")

    # Mostrar algunos valores originales para análisis
    sample_original = df['TimeAlive'].head(10).tolist()
    print("📋 Valores originales de muestra:")
    for i, val in enumerate(sample_original, 1):
        print(f"   {i:2d}. {val}")

    def convert_time_value(time_str):
        """Convierte valores de TimeAlive a formato utilizable"""
        try:
            if pd.isna(time_str):
                return np.nan

            # Si ya es numérico, retornarlo
            if isinstance(time_str, (int, float)):
                return float(time_str)

            # Convertir string removiendo puntos (separadores de miles)
            time_str = str(time_str).replace('.', '')
            time_numeric = float(time_str)

            # Probar diferentes escalas de conversión
            # Los valores parecen estar en una unidad de tiempo muy pequeña

            # Opción 1: Microsegundos (10^6 microsegundos = 1 segundo)
            time_microseconds = time_numeric / 1_000_000

            # Opción 2: Nanosegundos (10^9 nanosegundos = 1 segundo)
            time_nanoseconds = time_numeric / 1_000_000_000

            # Opción 3: Ticks de .NET (10^7 ticks = 1 segundo)
            time_ticks = time_numeric / 10_000_000

            # Analizar cuál conversión da valores más realistas para Counter Strike
            # Las rondas de CS duran típicamente 1-120 segundos

            # Retornar la conversión que da valores en rango realista
            if 0.1 <= time_ticks <= 300:  # Ticks parece más realista
                return time_ticks
            elif 0.1 <= time_microseconds <= 300:  # Microsegundos
                return time_microseconds
            elif 0.1 <= time_nanoseconds <= 300:  # Nanosegundos
                return time_nanoseconds
            else:
                # Si ninguna conversión da valores realistas, usar ticks como default
                return time_ticks

        except:
            return np.nan

    # Aplicar conversión
    df['TimeAlive_converted'] = df['TimeAlive'].apply(convert_time_value)

    # Mostrar estadísticas de diferentes conversiones para algunos valores
    print("\n📊 ANÁLISIS DE CONVERSIONES:")
    sample_vals = []
    for i in range(min(5, len(df))):
        original = df.iloc[i]['TimeAlive']
        converted = df.iloc[i]['TimeAlive_converted']

        try:
            # Calcular diferentes conversiones para comparar
            numeric_val = float(str(original).replace('.', ''))

            microseconds = numeric_val / 1_000_000
            nanoseconds = numeric_val / 1_000_000_000
            ticks = numeric_val / 10_000_000

            print(f"   Valor {i+1}: {original}")
            print(f"      Microsegundos: {microseconds:.2f}s")
            print(f"      Nanosegundos: {nanoseconds:.6f}s")
            print(f"      Ticks .NET: {ticks:.2f}s")
            print(f"      Conversión usada: {converted:.2f}s")

        except:
            continue

    return df


def train_models_final():
    print("🎮 ENTRENAMIENTO FINAL - ANÁLISIS COMPLETO DE TimeAlive")
    print("="*65)

    # 1. CARGAR DATOS
    print("📁 Cargando datos...")
    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
    print(
        f"✅ Datos cargados: {df.shape[0]:,} registros, {df.shape[1]} columnas")

    # 2. ANALIZAR Y CONVERTIR TimeAlive
    df = analyze_and_convert_time_alive(df)

    # 3. ESTADÍSTICAS DESPUÉS DE CONVERSIÓN
    valid_count = df['TimeAlive_converted'].notna().sum()
    print(f"\n📊 DESPUÉS DE CONVERSIÓN:")
    print(f"   Valores válidos: {valid_count:,}")

    if valid_count > 0:
        time_data = df['TimeAlive_converted'].dropna()
        print(f"   Mínimo: {time_data.min():.2f} segundos")
        print(f"   Máximo: {time_data.max():.2f} segundos")
        print(f"   Promedio: {time_data.mean():.2f} segundos")
        print(f"   Mediana: {time_data.median():.2f} segundos")

        # Analizar distribución
        realistic_count = ((time_data > 0.1) & (time_data <= 300)).sum()
        print(f"   Valores en rango realista (0.1-300s): {realistic_count:,}")

        # Si muy pocos valores son realistas, probar conversión diferente
        if realistic_count < valid_count * 0.1:  # Menos del 10% son realistas
            print("⚠️ Muy pocos valores realistas, probando conversión alternativa...")

            # Probar conversión más agresiva (dividir por un factor mayor)
            df['TimeAlive_alt'] = df['TimeAlive'].apply(lambda x:
                                                        float(str(x).replace('.', '')) / 100_000_000 if pd.notna(x) else np.nan)

            alt_time_data = df['TimeAlive_alt'].dropna()
            alt_realistic = ((alt_time_data > 0.1) &
                             (alt_time_data <= 300)).sum()

            print(
                f"   Conversión alternativa - valores realistas: {alt_realistic:,}")

            if alt_realistic > realistic_count:
                print("✅ Usando conversión alternativa")
                df['TimeAlive_converted'] = df['TimeAlive_alt']
                time_data = alt_time_data
                realistic_count = alt_realistic

    # 4. FILTRAR DATOS
    print(f"\n🧹 FILTRANDO DATOS...")
    initial_count = len(df)

    # Eliminar valores nulos
    df = df.dropna(subset=['TimeAlive_converted'])

    # Filtrar valores realistas para Counter Strike
    df = df[(df['TimeAlive_converted'] > 0.1) &
            (df['TimeAlive_converted'] <= 300)]

    final_count = len(df)
    print(f"   Registros iniciales: {initial_count:,}")
    print(f"   Registros finales: {final_count:,}")
    print(f"   Porcentaje conservado: {(final_count/initial_count)*100:.1f}%")

    if final_count < 1000:
        print(f"⚠️ ADVERTENCIA: Solo {final_count:,} registros válidos")

        # Si aún tenemos pocos datos, ser más permisivos
        df_permissive = pd.read_csv(
            'Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
        df_permissive = analyze_and_convert_time_alive(df_permissive)
        df_permissive = df_permissive.dropna(subset=['TimeAlive_converted'])
        df_permissive = df_permissive[(df_permissive['TimeAlive_converted'] > 0) &
                                      (df_permissive['TimeAlive_converted'] <= 1000)]  # Más permisivo

        if len(df_permissive) > final_count * 1.5:
            print(
                f"   Usando filtro más permisivo: {len(df_permissive):,} registros")
            df = df_permissive
            final_count = len(df)

    if final_count < 500:
        print(f"❌ ERROR: Muy pocos registros para entrenar ({final_count:,})")

        # Como último recurso, usar los datos "tal como están" con normalización
        print("🔄 Intentando usar datos sin conversión de tiempo...")
        df_raw = pd.read_csv(
            'Anexo ET_demo_round_traces_2022 (1).csv', sep=';')

        # Usar solo registros con TimeAlive numérico simple
        df_raw['TimeAlive_numeric'] = pd.to_numeric(
            df_raw['TimeAlive'], errors='coerce')
        df_raw = df_raw.dropna(subset=['TimeAlive_numeric'])
        df_raw = df_raw[df_raw['TimeAlive_numeric'] > 0]

        if len(df_raw) > 500:
            print(f"   Usando datos sin conversión: {len(df_raw):,} registros")
            df = df_raw
            df['TimeAlive_converted'] = df['TimeAlive_numeric']
            final_count = len(df)

    # 5. PREPARAR CARACTERÍSTICAS
    print(f"\n🔧 Preparando características...")

    # Codificar mapa
    if 'Map' in df.columns:
        le_map = LabelEncoder()
        df['Map_Encoded'] = le_map.fit_transform(df['Map'].fillna('Unknown'))
        print(f"   ✅ Mapa codificado: {df['Map'].nunique()} mapas únicos")
    else:
        df['Map_Encoded'] = 0

    # Características disponibles
    potential_features = [
        'MatchKills', 'MatchHeadshots', 'RoundKills', 'RoundHeadshots',
        'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue',
        'Map_Encoded', 'MatchAssists', 'RoundAssists'
    ]

    available_features = [f for f in potential_features if f in df.columns]
    print(f"   ✅ Características disponibles: {len(available_features)}")

    # 6. PREPARAR DATOS PARA ENTRENAMIENTO
    X = df[available_features].fillna(0)
    y_regression = df['TimeAlive_converted']

    # Variable de clasificación
    median_survival = y_regression.median()
    y_classification = (y_regression > median_survival).astype(int)

    print(f"\n📊 DATOS FINALES PARA ENTRENAMIENTO:")
    print(f"   Registros: {len(X):,}")
    print(f"   Características: {len(available_features)}")
    print(
        f"   TimeAlive rango: {y_regression.min():.2f} - {y_regression.max():.2f} segundos")
    print(f"   Mediana: {median_survival:.2f} segundos")
    print(
        f"   Distribución clasificación: {(y_classification == 0).sum():,} baja, {(y_classification == 1).sum():,} alta")

    # 7. ENTRENAR RANDOM FOREST
    print(f"\n🌳 ENTRENANDO RANDOM FOREST...")

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=100,  # Reducir para datasets pequeños
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_reg_train)
    y_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_reg_test, y_pred)

    print(f"   ✅ Random Forest R²: {rf_r2:.3f}")

    # 8. ENTRENAR XGBOOST
    print(f"\n🚀 ENTRENANDO XGBOOST...")

    X_train, X_test, y_class_train, y_class_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,  # Reducir para datasets pequeños
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )

    xgb_model.fit(X_train, y_class_train)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_class_test, y_proba)

    print(f"   ✅ XGBoost AUC: {xgb_auc:.3f}")

    # 9. GUARDAR MODELOS
    print(f"\n💾 Guardando modelos...")
    os.makedirs('backend/models', exist_ok=True)

    with open('backend/models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)

    with open('backend/models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)

    with open('backend/models/feature_names.pkl', 'wb') as f:
        pickle.dump(available_features, f)

    # 10. RESUMEN FINAL
    print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
    print("="*50)
    print(f"📊 Registros utilizados: {final_count:,}")
    print(f"🌳 Random Forest R²: {rf_r2:.3f}")
    print(f"🚀 XGBoost AUC: {xgb_auc:.3f}")

    if rf_r2 >= 0.3 or xgb_auc >= 0.7:
        print(f"✅ Rendimiento aceptable para continuar con la web")
    else:
        print(f"⚠️ Rendimiento bajo, pero modelos guardados para testing")

    print(f"\n🌐 PRÓXIMOS PASOS:")
    print(f"   cd backend && python app.py")

    return True


if __name__ == "__main__":
    train_models_final()
