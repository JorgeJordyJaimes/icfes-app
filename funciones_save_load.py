import pandas as pd
from rutas import data_dir

def guardar_parquet_en_data(df: pd.DataFrame, nombre_archivo: str = "icfes.parquet"):
    """
    Guarda un DataFrame en formato Parquet dentro de la carpeta 'data'.
    Usa la ruta relativa data_dir definida en rutas.py.
    """

    # Asegurar que la carpeta existe
    data_dir.mkdir(parents=True, exist_ok=True)

    # Construir ruta final
    ruta_parquet = data_dir / nombre_archivo

    # Guardar
    df.to_parquet(ruta_parquet)

    print(f"Archivo guardado en: {ruta_parquet}")
