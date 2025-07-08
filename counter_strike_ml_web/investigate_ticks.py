# =============================================================================
# INVESTIGACIÓN DEL FORMATO TICKS EN TimeAlive
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def analyze_ticks_format():
    print("🔍 ANÁLISIS DETALLADO DEL FORMATO TICKS")
    print("="*50)

    # Cargar datos
    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
    print(f"📊 Dataset cargado: {df.shape[0]:,} registros")

    # Tomar muestra de TimeAlive
    sample_values = df['TimeAlive'].head(20).tolist()

    print(f"\n📋 ANÁLISIS DE MUESTRA DE TimeAlive:")
    print("-" * 80)
    print(f"{'#':<3} {'Valor Original':<25} {'Sin Puntos':<20} {'Ticks→Seg':<12} {'Interpretación'}")
    print("-" * 80)

    for i, val in enumerate(sample_values, 1):
        try:
            # Remover puntos y convertir a número
            val_str = str(val).replace('.', '')
            val_numeric = int(val_str)

            # Diferentes interpretaciones de ticks
            # 1. Ticks .NET estándar (10^7 ticks = 1 segundo)
            ticks_standard = val_numeric / 10_000_000

            # 2. Ticks con precisión mayor (10^8 ticks = 1 segundo)
            ticks_high_precision = val_numeric / 100_000_000

            # 3. Ticks de sistema específico (10^6 ticks = 1 segundo)
            ticks_microseconds = val_numeric / 1_000_000

            # 4. Posible timestamp de época
            # Los valores grandes podrían ser timestamps desde epoch
            try:
                if val_numeric > 1e15:  # Si es muy grande, podría ser timestamp en nanosegundos
                    timestamp_ns = val_numeric
                    timestamp_s = timestamp_ns / 1e9
                    dt = datetime.fromtimestamp(timestamp_s)
                    interpretation = f"Timestamp: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
                elif ticks_high_precision < 300:  # Dentro de rango razonable para CS
                    interpretation = f"Ticks HP: {ticks_high_precision:.2f}s ✅"
                elif ticks_standard < 300:
                    interpretation = f"Ticks Std: {ticks_standard:.2f}s ✅"
                elif ticks_microseconds < 300:
                    interpretation = f"Microseg: {ticks_microseconds:.2f}s ✅"
                else:
                    interpretation = "Formato desconocido ❓"
            except:
                interpretation = "Error en conversión"

            print(
                f"{i:<3} {str(val):<25} {val_str:<20} {ticks_standard:<12.2f} {interpretation}")

        except Exception as e:
            print(f"{i:<3} {str(val):<25} ERROR: {str(e)}")

    # Análisis estadístico más profundo
    print(f"\n📊 ANÁLISIS ESTADÍSTICO PROFUNDO:")

    # Convertir todos los valores
    def convert_time_value(val, conversion_factor):
        try:
            val_str = str(val).replace('.', '')
            val_numeric = int(val_str)
            return val_numeric / conversion_factor
        except:
            return np.nan

    # Probar diferentes factores de conversión
    conversion_factors = {
        '10^6 (microsegundos)': 1_000_000,
        '10^7 (ticks .NET)': 10_000_000,
        '10^8 (ticks precisión alta)': 100_000_000,
        '10^9 (nanosegundos)': 1_000_000_000
    }

    print(f"\n🧪 PRUEBA DE DIFERENTES CONVERSIONES:")
    print("-" * 70)
    print(f"{'Factor':<25} {'Min':<8} {'Max':<12} {'Media':<10} {'En Rango CS':<12}")
    print("-" * 70)

    for name, factor in conversion_factors.items():
        converted = df['TimeAlive'].apply(
            lambda x: convert_time_value(x, factor))
        valid_converted = converted.dropna()

        if len(valid_converted) > 0:
            min_val = valid_converted.min()
            max_val = valid_converted.max()
            mean_val = valid_converted.mean()

            # Contar cuántos están en rango razonable para CS (0.1 - 300 segundos)
            in_range = ((valid_converted >= 0.1) &
                        (valid_converted <= 300)).sum()
            in_range_pct = (in_range / len(valid_converted)) * 100

            print(
                f"{name:<25} {min_val:<8.2f} {max_val:<12.1f} {mean_val:<10.1f} {in_range:,} ({in_range_pct:.1f}%)")

    # Análisis de patrones en los valores
    print(f"\n🔍 ANÁLISIS DE PATRONES:")

    # Verificar si hay relación con otras columnas
    if 'Survived' in df.columns:
        print(f"\n📊 RELACIÓN CON SUPERVIVENCIA:")
        survived_sample = df[['TimeAlive', 'Survived']].head(10)

        for idx, row in survived_sample.iterrows():
            time_val = row['TimeAlive']
            survived = row['Survived']

            # Convertir con el factor más prometedor
            try:
                val_numeric = int(str(time_val).replace('.', ''))
                time_seconds = val_numeric / 100_000_000  # Usando 10^8 como ejemplo

                status = "Sobrevivió" if survived else "Murió"
                print(
                    f"   TimeAlive: {time_val} → {time_seconds:.2f}s, {status}")
            except:
                continue

    # Análisis de FirstKillTime para comparación
    if 'FirstKillTime' in df.columns:
        print(f"\n⚔️ COMPARACIÓN CON FirstKillTime:")
        sample_kills = df[['TimeAlive', 'FirstKillTime']].dropna().head(5)

        for idx, row in sample_kills.iterrows():
            time_alive = row['TimeAlive']
            first_kill = row['FirstKillTime']

            print(f"   TimeAlive: {time_alive}")
            print(f"   FirstKillTime: {first_kill}")
            print()

    # Recomendación final
    print(f"\n💡 RECOMENDACIÓN BASADA EN ANÁLISIS:")

    # Usar el factor que da más valores en rango razonable
    best_factor = None
    best_count = 0

    for name, factor in conversion_factors.items():
        converted = df['TimeAlive'].apply(
            lambda x: convert_time_value(x, factor))
        valid_converted = converted.dropna()
        in_range = ((valid_converted >= 0.1) & (valid_converted <= 300)).sum()

        if in_range > best_count:
            best_count = in_range
            best_factor = (name, factor)

    if best_factor:
        print(f"✅ Mejor conversión: {best_factor[0]}")
        print(f"   Factor: {best_factor[1]:,}")
        print(f"   Registros en rango CS: {best_count:,}")

        # Mostrar código de implementación
        print(f"\n💻 CÓDIGO PARA IMPLEMENTAR:")
        print(f"```python")
        print(f"def convert_timealive_to_seconds(time_str):")
        print(f"    try:")
        print(f"        val_numeric = int(str(time_str).replace('.', ''))")
        print(f"        return val_numeric / {best_factor[1]}")
        print(f"    except:")
        print(f"        return np.nan")
        print(f"```")

    return best_factor


def test_conversion_with_survived():
    """
    Prueba la conversión correlacionando con la columna Survived
    """
    print(f"\n🧪 PRUEBA DE CORRELACIÓN CON SURVIVED")
    print("="*45)

    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')

    if 'Survived' not in df.columns:
        print("❌ Columna 'Survived' no encontrada")
        return

    # Probar diferentes conversiones y ver cuál correlaciona mejor con Survived
    conversion_factors = {
        'microsegundos': 1_000_000,
        'ticks_net': 10_000_000,
        'ticks_hp': 100_000_000,
        'nanosegundos': 1_000_000_000
    }

    def convert_time_value(val, factor):
        try:
            val_str = str(val).replace('.', '')
            val_numeric = int(val_str)
            return val_numeric / factor
        except:
            return np.nan

    print(f"📊 CORRELACIÓN CON SUPERVIVENCIA:")
    print("-" * 50)
    print(f"{'Conversión':<15} {'Correlación':<12} {'Promedio Sobrevivió':<18} {'Promedio Murió':<15}")
    print("-" * 50)

    for name, factor in conversion_factors.items():
        # Convertir TimeAlive
        df['TimeConverted'] = df['TimeAlive'].apply(
            lambda x: convert_time_value(x, factor))

        # Filtrar valores válidos y en rango razonable
        df_valid = df[(df['TimeConverted'].notna()) &
                      (df['TimeConverted'] >= 0.1) &
                      (df['TimeConverted'] <= 300)]

        if len(df_valid) > 100:  # Solo si tenemos suficientes datos
            # Calcular correlación
            correlation = df_valid['TimeConverted'].corr(
                df_valid['Survived'].astype(int))

            # Promedios por grupo
            avg_survived = df_valid[df_valid['Survived']
                                    == True]['TimeConverted'].mean()
            avg_died = df_valid[df_valid['Survived']
                                == False]['TimeConverted'].mean()

            print(
                f"{name:<15} {correlation:<12.3f} {avg_survived:<18.1f} {avg_died:<15.1f}")

    print(f"\n💡 La conversión con mayor correlación positiva es la más correcta")


if __name__ == "__main__":
    best_conversion = analyze_ticks_format()
    test_conversion_with_survived()
