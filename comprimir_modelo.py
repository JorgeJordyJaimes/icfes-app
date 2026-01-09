# Crea comprimir_modelo.py en C:\Users\CACTU\Downloads\Proyectos\icfes-app\
import joblib
import os

print("🔄 Comprimiendo modelo (esto puede tomar 1-2 minutos)...")

# Cargar modelo original
modelo = joblib.load('models/modelo_icfes_completo.pkl')

# Guardar COMPRIMIDO (nivel 9 = máxima compresión)
joblib.dump(modelo, 'models/modelo_comprimido.pkl', compress=9)

# Comparar tamaños
original_mb = os.path.getsize('models/modelo_icfes_completo.pkl') / 1024 / 1024
comprimido_mb = os.path.getsize('models/modelo_comprimido.pkl') / 1024 / 1024

print(f"📦 Original: {original_mb:.1f} MB")
print(f"📦 Comprimido: {comprimido_mb:.1f} MB")
print(f"📦 Reducción: {(1-comprimido_mb/original_mb)*100:.1f}%")

if comprimido_mb < 100:
    print("✅ ¡SUFICIENTEMENTE PEQUEÑO para GitHub!")
else:
    print("⚠️  Todavía muy grande, prueba Solución 2")