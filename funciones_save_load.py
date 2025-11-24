import pandas as pd
from rutas import data_dir


# ============================================================
# FUNCIÓN: GUARDAR ARCHIVO PARQUET EN LA CARPETA /data
# ============================================================
def guardar_parquet(df: pd.DataFrame, nombre_archivo: str = "archivo.parquet"):
    """
    Guarda un DataFrame en formato Parquet dentro de la carpeta 'data'.

    Parámetros:
        df (pd.DataFrame): DataFrame que deseas guardar.
        nombre_archivo (str): Nombre del archivo parquet a generar.
                              Ejemplo: 'icfes_2020_2.parquet'

    Retorna:
        ruta_parquet (Path): Ruta completa donde se guardó el archivo.
    """

    # Asegurar que la carpeta /data exista
    data_dir.mkdir(parents=True, exist_ok=True)

    # Ruta final del archivo
    ruta_parquet = data_dir / nombre_archivo

    # Guardar el DataFrame en formato Parquet
    df.to_parquet(ruta_parquet)

    print(f"Archivo guardado en: {ruta_parquet}")

    return ruta_parquet



# ============================================================
# FUNCIÓN: CARGAR ARCHIVO PARQUET DESDE LA CARPETA /data
# ============================================================
def cargar_parquet(nombre_archivo: str):
    """
    Carga un archivo Parquet ubicado en la carpeta 'data'.

    Parámetros:
        nombre_archivo (str): Nombre del archivo parquet a cargar.
                              Ejemplo: 'icfes_2020_2.parquet'

    Retorna:
        DataFrame: Contenido del archivo Parquet dentro de un DataFrame.
    """

    # Construir la ruta completa
    ruta_parquet = data_dir / nombre_archivo

    # Cargar el archivo
    df = pd.read_parquet(ruta_parquet)

    return df
