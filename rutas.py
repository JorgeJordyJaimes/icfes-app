from pathlib import Path


# Obtener la carpeta raíz del proyecto de forma robusta
# __file__ apunta al archivo rutas.py
carpeta_raiz = Path(__file__).resolve().parent

# Carpetas relativas de subcarpetas
data_dir = carpeta_raiz / "data"
app_dir = carpeta_raiz / "app"
notebooks_dir = carpeta_raiz / "notebooks"
modelos_dir = carpeta_raiz / "models"
