import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
import joblib
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path 

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
import joblib
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path 

# ========== CONFIGURACIÓN DE RUTAS ==========
# Obtener la ruta del directorio actual (donde está app_icfes.py)
current_dir = Path(__file__).resolve().parent

# Subir un nivel para llegar a la carpeta raíz del proyecto
project_root = current_dir.parent

# Ruta a la carpeta de modelos
models_dir = project_root / "models"

# Ruta completa al modelo
model_path = models_dir / "modelo_icfes_completo.pkl"

# Verificar que la ruta existe (para debugging)
print(f"📁 Directorio actual: {current_dir}")
print(f"📁 Raíz del proyecto: {project_root}")
print(f"📁 Carpeta modelos: {models_dir}")
print(f"📁 Ruta del modelo: {model_path}")
print(f"✅ Modelo existe: {model_path.exists()}")

# Configuración de la página
st.set_page_config(
    page_title="Predicción ICFES",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        # Usar la ruta configurada
        print(f"🔍 Intentando cargar modelo desde: {model_path}")
        
        if not model_path.exists():
            st.error(f"❌ El archivo del modelo no existe en: {model_path}")
            st.error(f"📁 Verifica que la carpeta 'models' contenga 'modelo_icfes_completo.pkl'")
            return None
        
        modelo = joblib.load(model_path)
        st.success("✅ Modelo cargado correctamente")
        print(f"✅ Modelo cargado exitosamente")
        
        # Opcional: mostrar información del modelo
        print(f"📊 Información del modelo:")
        print(f"   • Tipo: {type(modelo)}")
        print(f"   • Número de árboles: {getattr(modelo, 'n_estimators', 'No disponible')}")
        print(f"   • Número de características: {getattr(modelo, 'n_features_in_', 'No disponible')}")
        
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        print(f"❌ Error detallado: {e}")
        import traceback
        print(f"📋 Traceback completo:\n{traceback.format_exc()}")
        return None

def crear_formulario():
    """
    Crea un formulario con las variables originales
    """
    datos = {}
    
    st.title("📊 Predicción de Puntaje ICFES")
    st.markdown("Complete el siguiente formulario con información socioeconómica:")
    
    # ========== SECCIÓN 1: INFORMACIÓN PERSONAL ==========
    with st.expander("👤 INFORMACIÓN PERSONAL", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Edad (variable numérica)
            datos['edad'] = st.number_input(
                "Edad del estudiante",
                min_value=10,
                max_value=61,
                value=17,
                help="Edad en años"
            )
            
            # Nacionalidad
            datos['nacionalidad'] = st.selectbox(
                "Nacionalidad",
                options=['Colombia', 'Venezuela', 'Estados unidos', 'Ecuador', 'España',
                        'Cuba', 'Panamá', 'Brasil', 'Argentina', 'México', 'Perú',
                        'Vanuatu', 'Costa rica', 'Bolivia', 'Italia', 'Uruguay',
                        'Nicaragua', 'República dominicana', 'Honduras', 'Japón',
                        'Guatemala', 'Francia', 'Corea del sur', 'Cabo verde',
                        'El salvador', 'Otro'],
                index=0
            )
        
        with col2:
            # Pertenece a etnia
            datos['pertenece_etnia'] = st.selectbox(
                "¿Pertenece a alguna etnia?",
                options=['No', 'Si'],
                index=0
            )
            
            # Presentó fuera de edad
            datos['presento_fuera_edad'] = st.selectbox(
                "¿Presentó la prueba fuera de la edad esperada?",
                options=['0', '1'],  # 0=No, 1=Sí
                index=0,
                format_func=lambda x: "No" if x == "0" else "Sí"
            )
            
            # Región
            datos['region'] = st.selectbox(
                "Región de residencia",
                options=['Andina', 'Caribe', 'Pacífica', 'Orinoquía', 'Amazónica'],
                index=0
            )
    
    # ========== SECCIÓN 2: VIVIENDA Y HOGAR ==========
    with st.expander("🏠 CARACTERÍSTICAS DE LA VIVIEDA"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Estrato
            datos['estrato_casa'] = st.selectbox(
                "Estrato de la vivienda",
                options=['Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 
                        'Estrato 5', 'Estrato 6', 'Sin estrato'],
                index=1  # Estrato 2 es el más común
            )
        
        with col2:
            # Número de personas en la casa
            datos['num_personas_casa'] = st.selectbox(
                "Número de personas en el hogar",
                options=['1 a 2', '3 a 4', '5 a 6', '7 a 8', '9 o más'],
                index=1  # 3 a 4 es el más común
            )
        
        with col3:
            # Número de cuartos
            datos['cuartos_casa'] = st.selectbox(
                "Número de cuartos en la vivienda",
                options=['Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis o mas'],
                index=2  # Tres es el más común
            )
    
    # ========== SECCIÓN 3: RECURSOS DEL HOGAR ==========
    with st.expander("💻 RECURSOS Y BIENES"):
        st.write("¿Cuáles de estos recursos tiene en su hogar?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            datos['internet'] = st.checkbox("Internet", value=True)
            datos['tv'] = st.checkbox("Televisor", value=True)
            datos['computador'] = st.checkbox("Computador", value=True)
            datos['lavadora'] = st.checkbox("Lavadora", value=True)
        
        with col2:
            datos['microndas'] = st.checkbox("Microondas", value=False)
            datos['carro'] = st.checkbox("Carro", value=False)
            datos['moto'] = st.checkbox("Moto", value=False)
            datos['consola'] = st.checkbox("Consola de videojuegos", value=False)
    
    # ========== SECCIÓN 4: EDUCACIÓN FAMILIAR ==========
    with st.expander("🎓 EDUCACIÓN DE LOS PADRES"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Educación del padre (CORREGIDO)
            datos['educacion_padre'] = st.selectbox(
                "Nivel educativo del padre",
                options=['Bachiller', 'Primaria incompleta', 'Bachiller incompleta',
                        'Profesional', 'Primaria', 'Tecnico/Tecnologo', 'Ninguna',
                        'Profesional incompleta', 'Tecnico/Tecnologo incompleta', 'No aplica'],
                index=0  # Bachiller es el más común (96,325 casos)
            )
            
            # Actividad del padre
            datos['actividad_padre'] = st.selectbox(
                "Actividad económica del padre",
                options=['Trabajadores operativos', 'Microempresario', 'Sector primario',
                        'Trabajador independiente', 'Sin informacion',
                        'Profesionales', 'Sin actividad remunerada',
                        'Directivos', 'Pensionado'],
                index=0
            )
        
        with col2:
            # Educación de la madre (CORREGIDO)
            datos['educacion_madre'] = st.selectbox(
                "Nivel educativo de la madre",
                options=['Bachiller', 'Primaria incompleta', 'Bachiller incompleta',
                        'Profesional', 'Tecnico/Tecnologo', 'Primaria',
                        'Tecnico/Tecnologo incompleta', 'Profesional incompleta', 
                        'Ninguna', 'No aplica'],
                index=0  # Bachiller es el más común (110,294 casos)
            )
            
            # Actividad de la madre
            datos['actividad_madre'] = st.selectbox(
                "Actividad económica de la madre",
                options=['Sin actividad remunerada', 'Trabajadores operativos',
                        'Microempresario', 'Profesionales', 'Trabajador independiente',
                        'Sector primario', 'Sin informacion', 'Directivos', 'Pensionado'],
                index=0
            )
        
        # Educación padres (combinada) - CORREGIDO
        datos['educacion_padres'] = st.selectbox(
            "Nivel educativo combinado de los padres",
            options=['Al Menos Un Bachiller', 'Educación Superior', 'Educación Primaria Incompleta',
                    'Educación Técnica', 'Bachillerato Completo', 'Educación Primaria',
                    'Educación Secundaria Incompleta', 'Sin Información',
                    'Educación Superior Incompleta', 'No Aplica', 'Sin Educación Formal'],
            index=0  # Al Menos Un Bachiller es el más común (79,642 casos)
        )
    
    # ========== SECCIÓN 5: HÁBITOS Y COSTUMBRES ==========
    with st.expander("📚 HÁBITOS DE ESTUDIO Y LECTURA"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Número de libros
            datos['num_libros_casa'] = st.selectbox(
                "¿Cuántos libros hay en casa?",
                options=['0 a 10 libros', '11 a 25 libros', '26 a 100 libros', 'Más de 100 libros'],
                index=0
            )
            
            # Hábito de lectura
            datos['cuanto_lee'] = st.selectbox(
                "¿Cuánto lee por entretenimiento?",
                options=['30 minutos o menos', 'Entre 30 y 60 minutos',
                        'No leo por entretenimiento', 'Entre 1 y 2 horas', 'Más de 2 horas'],
                index=0
            )
            
            # Perfil lector
            datos['perfil_lector'] = st.selectbox(
                "Perfil de lector",
                options=['Poco Apoyo, Poco Habito', 'Poco Apoyo, Buen Habito',
                        'Buen Apoyo, Buen Habito', 'Buen Apoyo, Poco Habito', 'Desconocido'],
                index=0
            )
        
        with col2:
            # Navegación web
            datos['cuanto_navega_web'] = st.selectbox(
                "¿Cuánto navega en internet?",
                options=['Entre 1 y 3 horas', 'Más de 3 horas', 'Entre 30 y 60 minutos',
                        '30 minutos o menos', 'No navega internet'],
                index=0
            )
            
            # Horas de trabajo
            datos['horas_trabajo_semanal'] = st.selectbox(
                "Horas de trabajo semanal",
                options=['No trabaja', 'Trabajo ocasional', 'Tiempo parcial reducido',
                        'Tiempo completo', 'Medio tiempo'],
                index=0
            )
    
    # ========== SECCIÓN 6: HÁBITOS ALIMENTICIOS ==========
    with st.expander("🍽️ HÁBITOS ALIMENTICIOS"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            datos['come_derivados_leche'] = st.selectbox(
                "Frecuencia de consumo de derivados de leche",
                options=['1 o 2 veces por semana', 'Todos o casi todos los días',
                        '3 a 5 veces por semana', 'Nunca o rara vez comemos eso'],
                index=0
            )
        
        with col2:
            datos['come_carne_pescado_huevo'] = st.selectbox(
                "Frecuencia de consumo de carne, pescado o huevo",
                options=['Todos o casi todos los días', '3 a 5 veces por semana',
                        '1 o 2 veces por semana', 'Nunca o rara vez comemos eso'],
                index=0
            )
        
        with col3:
            datos['come_cereal_frutas_legumbres'] = st.selectbox(
                "Frecuencia de consumo de cereales, frutas o legumbres",
                options=['1 o 2 veces por semana', '3 a 5 veces por semana',
                        'Nunca o rara vez comemos eso', 'Todos o casi todos los días'],
                index=0
            )
    
    # ========== SECCIÓN 7: INFORMACIÓN DEL COLEGIO ==========
    with st.expander("🏫 INFORMACIÓN DEL COLEGIO"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            datos['colegio_genero'] = st.selectbox(
                "Género del colegio",
                options=['Mixto', 'Femenino', 'Masculino'],
                index=0
            )
            
            datos['colegio_oficial'] = st.selectbox(
                "¿El colegio es oficial?",
                options=['Oficial', 'No oficial'],
                index=0
            )
        
        with col2:
            datos['colegio_bilingue'] = st.selectbox(
                "¿El colegio es bilingüe?",
                options=['No', 'Si'],
                index=0
            )
            
            datos['tipo_colegio'] = st.selectbox(
                "Tipo de colegio",
                options=['Académico', 'Técnico/académico', 'Técnico', 'No aplica'],
                index=0
            )
        
        with col3:
            datos['colegio_urbano_rural'] = st.selectbox(
                "Ubicación del colegio",
                options=['Urbano', 'Rural'],
                index=0
            )
            
            datos['colegio_jornada'] = st.selectbox(
                "Jornada del colegio",
                options=['Mañana', 'Unica', 'Completa', 'Tarde', 'Sabatina', 'Noche'],
                index=0
            )
    
    return datos

def transformar_a_one_hot(datos_usuario):
    """
    Transforma los datos del formulario al formato one-hot que espera el modelo.
    
    Args:
        datos_usuario: Diccionario con las respuestas del formulario
    
    Returns:
        DataFrame con 128 columnas en formato one-hot
    """
    # Lista de todas las columnas que espera el modelo (ACTUALIZADA)
    columnas_modelo = [
        'nacionalidad', 'pertenece_etnia', 'internet', 'tv', 'computador',
        'lavadora', 'microndas', 'carro', 'moto', 'consola',
        'colegio_bilingue', 'edad', 'presento_fuera_edad',
        'estrato_casa_estrato 1', 'estrato_casa_estrato 2',
        'estrato_casa_estrato 3', 'estrato_casa_estrato 4',
        'estrato_casa_estrato 5', 'estrato_casa_estrato 6',
        'estrato_casa_sin estrato', 'estrato_casa_nan',
        'num_personas_casa_1 a 2', 'num_personas_casa_3 a 4',
        'num_personas_casa_5 a 6', 'num_personas_casa_7 a 8',
        'num_personas_casa_9 o más', 'num_personas_casa_nan',
        'cuartos_casa_cinco', 'cuartos_casa_cuatro', 'cuartos_casa_dos',
        'cuartos_casa_seis o mas', 'cuartos_casa_tres', 'cuartos_casa_uno',
        'cuartos_casa_nan', 
        # Educación del padre (CORREGIDO)
        'educacion_padre_bachiller',
        'educacion_padre_bachiller incompleta', 'educacion_padre_ninguna',
        'educacion_padre_no aplica', 'educacion_padre_primaria',
        'educacion_padre_primaria incompleta',
        'educacion_padre_profesional',
        'educacion_padre_profesional incompleta',
        'educacion_padre_tecnico/tecnologo',
        'educacion_padre_tecnico/tecnologo incompleta',
        'educacion_padre_nan', 
        # Educación de la madre (CORREGIDO)
        'educacion_madre_bachiller',
        'educacion_madre_bachiller incompleta', 'educacion_madre_ninguna',
        'educacion_madre_no aplica', 'educacion_madre_primaria',
        'educacion_madre_primaria incompleta',
        'educacion_madre_profesional',
        'educacion_madre_profesional incompleta',
        'educacion_madre_tecnico/tecnologo',
        'educacion_madre_tecnico/tecnologo incompleta',
        'educacion_madre_nan', 
        'actividad_padre_directivos',
        'actividad_padre_microempresario', 'actividad_padre_pensionado',
        'actividad_padre_profesionales', 'actividad_padre_sector primario',
        'actividad_padre_sin actividad remunerada',
        'actividad_padre_sin informacion',
        'actividad_padre_trabajador independiente',
        'actividad_padre_trabajadores operativos', 'actividad_padre_nan',
        'actividad_madre_directivos', 'actividad_madre_microempresario',
        'actividad_madre_pensionado', 'actividad_madre_profesionales',
        'actividad_madre_sector primario',
        'actividad_madre_sin actividad remunerada',
        'actividad_madre_sin informacion',
        'actividad_madre_trabajador independiente',
        'actividad_madre_trabajadores operativos', 'actividad_madre_nan',
        'num_libros_casa_0 a 10 libros', 'num_libros_casa_11 a 25 libros',
        'num_libros_casa_26 a 100 libros',
        'num_libros_casa_más de 100 libros', 'num_libros_casa_nan',
        'come_derivados_leche_1 o 2 veces por semana',
        'come_derivados_leche_3 a 5 veces por semana',
        'come_derivados_leche_nunca o rara vez comemos eso',
        'come_derivados_leche_todos o casi todos los días',
        'come_derivados_leche_nan',
        'come_carne_pescado_huevo_1 o 2 veces por semana',
        'come_carne_pescado_huevo_3 a 5 veces por semana',
        'come_carne_pescado_huevo_nunca o rara vez comemos eso',
        'come_carne_pescado_huevo_todos o casi todos los días',
        'come_carne_pescado_huevo_nan',
        'come_cereal_frutas_legumbres_1 o 2 veces por semana',
        'come_cereal_frutas_legumbres_3 a 5 veces por semana',
        'come_cereal_frutas_legumbres_nunca o rara vez comemos eso',
        'come_cereal_frutas_legumbres_todos o casi todos los días',
        'come_cereal_frutas_legumbres_nan',
        'cuanto_lee_30 minutos o menos', 'cuanto_lee_entre 1 y 2 horas',
        'cuanto_lee_entre 30 y 60 minutos', 'cuanto_lee_más de 2 horas',
        'cuanto_lee_no leo por entretenimiento', 'cuanto_lee_nan',
        'cuanto_navega_web_30 minutos o menos',
        'cuanto_navega_web_entre 1 y 3 horas',
        'cuanto_navega_web_entre 30 y 60 minutos',
        'cuanto_navega_web_más de 3 horas',
        'cuanto_navega_web_no navega internet', 'cuanto_navega_web_nan',
        'horas_trabajo_semanal_medio tiempo',
        'horas_trabajo_semanal_no trabaja',
        'horas_trabajo_semanal_tiempo completo',
        'horas_trabajo_semanal_tiempo parcial reducido',
        'horas_trabajo_semanal_trabajo ocasional',
        'horas_trabajo_semanal_nan', 'colegio_genero_femenino',
        'colegio_genero_masculino', 'colegio_genero_mixto',
        'colegio_genero_nan', 'colegio_oficial_no oficial',
        'colegio_oficial_oficial', 'colegio_oficial_nan',
        'tipo_colegio_académico', 'tipo_colegio_no aplica',
        'tipo_colegio_técnico', 'tipo_colegio_técnico/académico',
        'tipo_colegio_nan', 'colegio_urbano_rural_rural',
        'colegio_urbano_rural_urbano', 'colegio_urbano_rural_nan',
        'colegio_jornada_completa', 'colegio_jornada_mañana',
        'colegio_jornada_noche', 'colegio_jornada_sabatina',
        'colegio_jornada_tarde', 'colegio_jornada_unica',
        'colegio_jornada_nan', 
        # Educación padres combinada (CORREGIDO)
        'educacion_padres_al menos un bachiller',
        'educacion_padres_bachillerato completo',
        'educacion_padres_educación primaria',
        'educacion_padres_educación primaria incompleta',
        'educacion_padres_educación secundaria incompleta',
        'educacion_padres_educación superior',
        'educacion_padres_educación superior incompleta',
        'educacion_padres_educación técnica', 'educacion_padres_no aplica',
        'educacion_padres_sin educación formal',
        'educacion_padres_sin información', 'educacion_padres_nan',
        'perfil_lector_buen apoyo, buen habito',
        'perfil_lector_buen apoyo, poco habito',
        'perfil_lector_desconocido',
        'perfil_lector_poco apoyo, buen habito',
        'perfil_lector_poco apoyo, poco habito', 'perfil_lector_nan',
        'region_amazónica', 'region_andina', 'region_caribe',
        'region_orinoquía', 'region_pacífica', 'region_nan'
    ]
    
    # Crear DataFrame con ceros
    df_one_hot = pd.DataFrame(0, index=[0], columns=columnas_modelo)
    
    # ========== MAPEO DE VARIABLES ==========
    
    # 1. Variables binarias directas
    binarias = ['internet', 'tv', 'computador', 'lavadora', 'microndas',
               'carro', 'moto', 'consola', 'colegio_bilingue']
    
    for var in binarias:
        if var in datos_usuario:
            if isinstance(datos_usuario[var], bool):
                df_one_hot[var] = 1 if datos_usuario[var] else 0
            elif isinstance(datos_usuario[var], str):
                df_one_hot[var] = 1 if datos_usuario[var].lower() in ['si', 'sí', 'yes', 'true', '1'] else 0
    
    # 2. Variables numéricas directas
    if 'edad' in datos_usuario:
        df_one_hot['edad'] = datos_usuario['edad']
    
    if 'presento_fuera_edad' in datos_usuario:
        df_one_hot['presento_fuera_edad'] = int(datos_usuario['presento_fuera_edad'])
    
    # 3. Variables categóricas (one-hot encoding manual) - ACTUALIZADO
    mapeo_categorias = {
        # Estrato
        'estrato_casa': {
            'Estrato 1': 'estrato_casa_estrato 1',
            'Estrato 2': 'estrato_casa_estrato 2',
            'Estrato 3': 'estrato_casa_estrato 3',
            'Estrato 4': 'estrato_casa_estrato 4',
            'Estrato 5': 'estrato_casa_estrato 5',
            'Estrato 6': 'estrato_casa_estrato 6',
            'Sin estrato': 'estrato_casa_sin estrato'
        },
        
        # Número de personas
        'num_personas_casa': {
            '1 a 2': 'num_personas_casa_1 a 2',
            '3 a 4': 'num_personas_casa_3 a 4',
            '5 a 6': 'num_personas_casa_5 a 6',
            '7 a 8': 'num_personas_casa_7 a 8',
            '9 o más': 'num_personas_casa_9 o más'
        },
        
        # Cuartos
        'cuartos_casa': {
            'Uno': 'cuartos_casa_uno',
            'Dos': 'cuartos_casa_dos',
            'Tres': 'cuartos_casa_tres',
            'Cuatro': 'cuartos_casa_cuatro',
            'Cinco': 'cuartos_casa_cinco',
            'Seis o mas': 'cuartos_casa_seis o mas'
        },
        
        # Región
        'region': {
            'Amazónica': 'region_amazónica',
            'Andina': 'region_andina',
            'Caribe': 'region_caribe',
            'Orinoquía': 'region_orinoquía',
            'Pacífica': 'region_pacífica'
        },
        
        # Educación del padre - NUEVO
        'educacion_padre': {
            'Bachiller': 'educacion_padre_bachiller',
            'Bachiller incompleta': 'educacion_padre_bachiller incompleta',
            'Ninguna': 'educacion_padre_ninguna',
            'No aplica': 'educacion_padre_no aplica',
            'Primaria': 'educacion_padre_primaria',
            'Primaria incompleta': 'educacion_padre_primaria incompleta',
            'Profesional': 'educacion_padre_profesional',
            'Profesional incompleta': 'educacion_padre_profesional incompleta',
            'Tecnico/Tecnologo': 'educacion_padre_tecnico/tecnologo',
            'Tecnico/Tecnologo incompleta': 'educacion_padre_tecnico/tecnologo incompleta'
        },
        
        # Educación de la madre - NUEVO
        'educacion_madre': {
            'Bachiller': 'educacion_madre_bachiller',
            'Bachiller incompleta': 'educacion_madre_bachiller incompleta',
            'Ninguna': 'educacion_madre_ninguna',
            'No aplica': 'educacion_madre_no aplica',
            'Primaria': 'educacion_madre_primaria',
            'Primaria incompleta': 'educacion_madre_primaria incompleta',
            'Profesional': 'educacion_madre_profesional',
            'Profesional incompleta': 'educacion_madre_profesional incompleta',
            'Tecnico/Tecnologo': 'educacion_madre_tecnico/tecnologo',
            'Tecnico/Tecnologo incompleta': 'educacion_madre_tecnico/tecnologo incompleta'
        },
        
        # Educación padres combinada - NUEVO
        'educacion_padres': {
            'Al Menos Un Bachiller': 'educacion_padres_al menos un bachiller',
            'Bachillerato Completo': 'educacion_padres_bachillerato completo',
            'Educación Primaria': 'educacion_padres_educación primaria',
            'Educación Primaria Incompleta': 'educacion_padres_educación primaria incompleta',
            'Educación Secundaria Incompleta': 'educacion_padres_educación secundaria incompleta',
            'Educación Superior': 'educacion_padres_educación superior',
            'Educación Superior Incompleta': 'educacion_padres_educación superior incompleta',
            'Educación Técnica': 'educacion_padres_educación técnica',
            'No Aplica': 'educacion_padres_no aplica',
            'Sin Educación Formal': 'educacion_padres_sin educación formal',
            'Sin Información': 'educacion_padres_sin información'
        }
    }
    
    # Aplicar mapeo para variables categóricas conocidas
    for var_original, mapeo in mapeo_categorias.items():
        if var_original in datos_usuario:
            valor = datos_usuario[var_original]
            if valor in mapeo:
                columna_one_hot = mapeo[valor]
                if columna_one_hot in df_one_hot.columns:
                    df_one_hot[columna_one_hot] = 1
    
    # 4. Para variables no mapeadas explícitamente, usar búsqueda inteligente
    variables_no_mapeadas = [var for var in datos_usuario.keys() 
                            if var not in binarias 
                            and var not in ['edad', 'presento_fuera_edad']
                            and var not in mapeo_categorias]
    
    for var_original in variables_no_mapeadas:
        valor = datos_usuario[var_original]
        if isinstance(valor, str):
            valor_limpio = valor.lower().replace(' ', '_').replace('/', '_').replace('ó', 'o')
            
            # Buscar coincidencia exacta primero
            encontrado = False
            for columna in df_one_hot.columns:
                # Verificar si la variable y valor están en la columna
                var_en_columna = var_original.lower().replace('_', ' ') in columna.lower()
                valor_en_columna = valor_limpio in columna.lower().replace(' ', '_')
                
                if var_en_columna and valor_en_columna:
                    df_one_hot[columna] = 1
                    encontrado = True
                    break
            
            # Si no se encontró, buscar coincidencia parcial
            if not encontrado:
                for columna in df_one_hot.columns:
                    if var_original in columna:
                        # Intentar coincidencia parcial del valor
                        palabras_valor = valor.lower().split()
                        for palabra in palabras_valor:
                            if palabra in columna.lower() and len(palabra) > 2:
                                df_one_hot[columna] = 1
                                encontrado = True
                                break
                    if encontrado:
                        break
    
    return df_one_hot

def main():
    """Función principal de la aplicación"""
    
    # Cargar modelo
    modelo = cargar_modelo()
    
    if modelo is None:
        st.error("No se pudo cargar el modelo. Verifica la ruta del archivo.")
        st.stop()
    
    # Crear formulario
    st.sidebar.title("🔍 Navegación")
    pagina = st.sidebar.radio(
        "Selecciona una página:",
        ["🏠 Inicio", "🎯 Predicción", "📊 Análisis"]
    )
    
    if pagina == "🏠 Inicio":
        st.title("Bienvenido al Predictor de Puntajes ICFES")
        st.write("""
        Esta herramienta utiliza un modelo de Random Forest para predecir puntajes 
        del ICFES basado en variables socioeconómicas.
        
        **Características:**
        - Predicción individual de puntajes
        - Análisis de importancia de variables
        - Visualización de resultados
        
        **Instrucciones:**
        1. Navega a la pestaña **Predicción**
        2. Completa el formulario con la información requerida
        3. Haz clic en **Predecir Puntaje**
        4. Revisa los resultados y análisis
        """)
    
    elif pagina == "🎯 Predicción":
        # Crear formulario
        datos_usuario = crear_formulario()
        
        st.markdown("---")
        
        # Botón para predecir
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Predecir Puntaje ICFES", type="primary", use_container_width=True):
                with st.spinner("Procesando datos y calculando predicción..."):
                    try:
                        # Transformar datos a one-hot
                        datos_one_hot = transformar_a_one_hot(datos_usuario)
                        
                        # Hacer predicción
                        prediccion = modelo.predict(datos_one_hot)
                        
                        # Mostrar resultado
                        st.success("✅ Predicción completada")
                        
                        st.markdown("---")
                        st.markdown(f"### 📊 Puntaje Predicho: **{prediccion[0]:.0f} puntos**")
                        
                        # Interpretación del resultado
                        with st.expander("📈 Interpretación del resultado"):
                            st.write(f"""
                            **Puntaje ICFES predicho:** {prediccion[0]:.0f} puntos
                            
                            **Interpretación:**
                            - Este valor representa el puntaje esperado en la prueba ICFES
                            - El rango típico de puntajes va de 0 a 500 puntos
                            - Factores que más influyeron en esta predicción:
                              1. Variable A
                              2. Variable B
                              3. Variable C
                            
                            **Nota:** Esta es una estimación basada en patrones históricos.
                            El resultado real puede variar.
                            """)
                        
                        # Mostrar datos procesados (opcional, para debugging)
                        with st.expander("🔍 Ver datos procesados (para desarrollo)"):
                            st.write("**Datos en formato one-hot:**")
                            st.dataframe(datos_one_hot.T[datos_one_hot.T[0] > 0])
                            
                    except Exception as e:
                        st.error(f"❌ Error al hacer la predicción: {str(e)}")
                        st.info("""
                        **Posibles soluciones:**
                        1. Verifica que todos los campos estén completos
                        2. Asegúrate de que el modelo esté correctamente cargado
                        3. Revisa la transformación de datos a one-hot
                        """)
    
    elif pagina == "📊 Análisis":
        st.title("📊 Análisis del Modelo")
        st.write("En construcción...")
        # Aquí irán las métricas, gráficos de importancia, etc.

if __name__ == "__main__":
    main()