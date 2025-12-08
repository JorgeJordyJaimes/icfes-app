import pandas as pd
from rutas import data_dir
from rutas import modelos_dir
import joblib


# ============================================================
# FUNCIÓN: GUARDAR ARCHIVO PARQUET EN LA CARPETA /data
# ============================================================
def guardar_parquet(df: pd.DataFrame, nombre_archivo: str = "archivo.parquet"):
    """
    Guarda un DataFrame o Series en formato Parquet dentro de la carpeta 'data'.

    Parámetros:
        df (pd.DataFrame o pd.Series): Objeto a guardar.
        nombre_archivo (str): Nombre del archivo parquet a generar.
                              Ejemplo: 'icfes_2020_2.parquet'

    Retorna:
        ruta_parquet (Path): Ruta completa donde se guardó el archivo.
    """

    # Si es una Series → convertir a DataFrame
    if isinstance(df, pd.Series):
        # Si la Series no tiene nombre, asignamos uno
        col_name = df.name if df.name is not None else "col_0"
        df = df.to_frame(name=col_name)

    # Asegurar que la carpeta /data exista
    data_dir.mkdir(parents=True, exist_ok=True)

    # Ruta final del archivo
    ruta_parquet = data_dir / nombre_archivo

    # Guardar el archivo
    df.to_parquet(ruta_parquet, index=False)

    return print(f'Archivo guardado en la carpeta data con el nombre de {nombre_archivo}')




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


# ============================================================
# GUARDAR MODELO
# ============================================================

def guardar_modelo(modelo, nombre_archivo="modelo_icfes.pkl"):
    
    modelos_ruta = modelos_dir / nombre_archivo
    
    joblib.dump(modelo, modelos_ruta)
    
    print(f"Modelo guardado con el nombre de: {nombre_archivo}")


# ============================================================
# CARGAR MODELO
# ============================================================

def cargar_modelo(nombre_archivo="modelo_icfes.pkl"):
    
    modelos_ruta = modelos_dir / nombre_archivo

    if not modelos_ruta.exists():
        raise FileNotFoundError(
            f"El modelo '{nombre_archivo}' no se encontró en {modelos_dir}"
        )
    
    print(f"Cargado modelo: {nombre_archivo}")
    return joblib.load(modelos_ruta)