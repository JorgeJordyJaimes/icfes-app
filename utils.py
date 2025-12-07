import pandas as pd
import numpy as np

def codificar_binarias(df, columnas, valor_si="Si"):

    for col in columnas:
        df[col] = df[col].apply(
            lambda x: 1 if x == valor_si else (0 if pd.notna(x) else np.nan)
        )



def codificar_one_hot_minusculas(df, columnas, dummy_na=True):
    """
    Aplica One-Hot Encoding a las columnas indicadas.
    Convierte los nombres de las categorías a minúsculas, mantiene todas las categorías
    y garantiza que las variables creadas sean numéricas (0/1).
    """

    for col in columnas:

        dummies = pd.get_dummies(
            df[col],
            prefix=col.lower(),
            prefix_sep="_",
            dummy_na=dummy_na
        )

        # Todo en minúsculas
        dummies.columns = dummies.columns.str.lower()

        # Convertir booleanos a 1/0
        dummies = dummies.astype(int)

        # Reemplazar columna original
        df.drop(columns=[col], inplace=True)
        df[dummies.columns] = dummies
