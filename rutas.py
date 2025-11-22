from pathlib import Path

# Carpeta raíz del proyecto
carpeta_raiz = Path.cwd()
print(carpeta_raiz)

# Carpetas relativas de subcarpetas
app_dir = carpeta_raiz / "app"
data_dir = carpeta_raiz / "data"
notebooks_dir = carpeta_raiz / "notebooks"
modelos_dir = carpeta_raiz / "models"
