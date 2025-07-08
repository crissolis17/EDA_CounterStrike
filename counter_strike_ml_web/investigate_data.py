# =============================================================================
# INVESTIGAR DATOS DE TimeAlive
# =============================================================================

import pandas as pd
import numpy as np


def investigate_time_alive():
    print("🔍 INVESTIGANDO DATOS DE TimeAlive")
    print("="*40)

    # Cargar datos
    df = pd.read_csv('Anexo ET_demo_round_traces_2022 (1).csv', sep=';')
    print(f"📊 Dataset cargado: {df.shape[0]:,} registros")

    # Investigar TimeAlive original
    print(f"\n📋 ANÁLISIS DE TimeAlive ORIGINAL:")
    print(f"   Tipo de datos: {df['TimeAlive'].dtype}")
    print(f"   Valores únicos: {df['TimeAlive'].nunique()}")
    print(f"   Valores nulos: {df['TimeAlive'].isnull().sum():,}")

    # Mostrar muestra de valores
    print(f"\n📋 MUESTRA DE VALORES TimeAlive:")
    sample_values = df['TimeAlive'].head(20).tolist()
    for i, val in enumerate(sample_values, 1):
        print(f"   {i:2d}. {val}")

    # Intentar conversión a numérico
    print(f"\n🔄 CONVIRTIENDO A NUMÉRICO:")
    df['TimeAlive_numeric'] = pd.to_numeric(df['TimeAlive'], errors='coerce')

    valid_numeric = df['TimeAlive_numeric'].notna().sum()
    invalid_numeric = df['TimeAlive_numeric'].isna().sum()

    print(f"   Valores válidos después de conversión: {valid_numeric:,}")
    print(f"   Valores inválidos: {invalid_numeric:,}")

    if valid_numeric > 0:
        print(f"\n📊 ESTADÍSTICAS DE VALORES VÁLIDOS:")
        valid_data = df['TimeAlive_numeric'].dropna()
        print(f"   Mínimo: {valid_data.min()}")
        print(f"   Máximo: {valid_data.max()}")
        print(f"   Promedio: {valid_data.mean():.2f}")
        print(f"   Mediana: {valid_data.median():.2f}")
        print(f"   Percentil 25: {valid_data.quantile(0.25):.2f}")
        print(f"   Percentil 75: {valid_data.quantile(0.75):.2f}")
        print(f"   Percentil 95: {valid_data.quantile(0.95):.2f}")
        print(f"   Percentil 99: {valid_data.quantile(0.99):.2f}")

        # Analizar distribución por rangos
        print(f"\n📈 DISTRIBUCIÓN POR RANGOS:")
        ranges = [
            (0, 1, "0-1 segundos"),
            (1, 10, "1-10 segundos"),
            (10, 30, "10-30 segundos"),
            (30, 60, "30-60 segundos"),
            (60, 120, "1-2 minutos"),
            (120, 300, "2-5 minutos"),
            (300, 600, "5-10 minutos"),
            (600, float('inf'), "> 10 minutos")
        ]

        for min_val, max_val, label in ranges:
            count = ((valid_data >= min_val) & (valid_data < max_val)).sum()
            percentage = (count / len(valid_data)) * 100
            print(f"   {label:<15}: {count:,} registros ({percentage:.1f}%)")

        # Filtros sugeridos
        print(f"\n💡 FILTROS SUGERIDOS:")

        # Filtro actual (muy restrictivo)
        current_filter = ((valid_data > 0) & (valid_data < 500)).sum()
        print(
            f"   Filtro actual (0 < TimeAlive < 500s): {current_filter:,} registros")

        # Filtros alternativos
        alt_filters = [
            (0, 1000, "0 < TimeAlive < 1000s"),
            (0, 2000, "0 < TimeAlive < 2000s"),
            (0, float('inf'), "TimeAlive > 0 (sin límite superior)")
        ]

        for min_val, max_val, label in alt_filters:
            if max_val == float('inf'):
                count = (valid_data > min_val).sum()
            else:
                count = ((valid_data > min_val) & (valid_data < max_val)).sum()
            percentage = (count / len(df)) * 100
            print(
                f"   {label}: {count:,} registros ({percentage:.1f}% del total)")

    # Analizar valores no numéricos
    if invalid_numeric > 0:
        print(f"\n🔍 ANÁLISIS DE VALORES NO NUMÉRICOS:")
        non_numeric = df[df['TimeAlive_numeric'].isna(
        )]['TimeAlive'].value_counts().head(10)
        print(f"   Top 10 valores no numéricos:")
        for val, count in non_numeric.items():
            print(f"      '{val}': {count:,} registros")


if __name__ == "__main__":
    investigate_time_alive()
