import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
import joblib
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import plotly.graph_objects as go

# ========== CONFIGURACIÓN DE RUTAS ==========
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
models_dir = project_root / "models"
model_path = models_dir / "modelo_comprimido.pkl"


# Configuración de la página
st.set_page_config(
    page_title="Predicción ICFES",
    page_icon="📊",
    layout="wide"
)

# Constantes para métricas del modelo
MAE_TEST = 31.297
PROMEDIO_ICFES = 248.0
DESV_STD_ICFES = 48.50

# PALETA DE COLORES PROFESIONAL
COLORES = {
    'primario': '#4A90E2',        # Azul principal
    'secundario': '#7B68EE',      # Púrpura
    'exito': '#50C878',           # Verde esmeralda
    'advertencia': '#FFB347',     # Naranja suave
    'error': '#FF6B6B',           # Rojo coral
    'info': '#48D1CC',            # Turquesa
    'fondo_claro': '#F8F9FA',     # Gris muy claro
    'texto_oscuro': '#2C3E50',    # Azul oscuro
    'degradado_1': '#667EEA',     # Inicio degradado
    'degradado_2': '#764BA2',     # Fin degradado
}

# CSS personalizado mejorado
st.markdown(f"""
<style>
    .stMetric {{
        background: linear-gradient(135deg, {COLORES['primario']}15 0%, {COLORES['secundario']}15 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid {COLORES['primario']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }}
    
    .main-header {{
        font-size: 2.5rem;
        color: {COLORES['primario']};
        font-weight: 700;
        margin-bottom: 1rem;
    }}
    
    .highlight {{
        background: linear-gradient(120deg, {COLORES['primario']}30 0%, {COLORES['info']}30 100%);
        padding: 15px;
        border-radius: 8px;
        font-weight: 600;
    }}
    
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-20px); }}
        100% {{ transform: translateY(0px); }}
    }}
    .balloon {{
        font-size: 3rem;
        animation: float 3s ease-in-out infinite;
        display: inline-block;
        margin: 0 10px;
    }}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado"""
    try:
        print(f"🔍 Intentando cargar modelo desde: {model_path}")
        
        if not model_path.exists():
            st.error(f"❌ El archivo del modelo no existe en: {model_path}")
            st.error(f"📁 Verifica que la carpeta 'models' contenga 'modelo_icfes_completo.pkl'")
            return None
        
        modelo = joblib.load(model_path)
        st.success("✅ Modelo cargado correctamente")
        print(f"✅ Modelo cargado exitosamente")
        
        return modelo
    except Exception as e:
        st.error(f"❌ Error cargando el modelo: {e}")
        print(f"❌ Error detallado: {e}")
        import traceback
        print(f"📋 Traceback completo:\n{traceback.format_exc()}")
        return None

def crear_formulario():
    """Crea un formulario con las variables originales"""
    datos = {}
    
    st.title("📊 Predicción de Puntaje ICFES")
    st.markdown("""El objetivo de este ejercicio es proponer un pequeño “juego” al usuario.
Muchas personas suelen afirmar que el examen ICFES “antes era más difícil”. La invitación es a que te pongas en los zapatos de tu yo del pasado y respondas el formulario según tu situación socioeconómica real en la época en que presentaste el examen.

Con esa información, el modelo estimará cómo te habría ido si hubieras presentado el ICFES con esas mismas condiciones, pero en el examen actual. El resultado no es una predicción exacta, sino una forma interesante de comparar percepciones y reflexionar sobre qué tan diferente (o no) es el examen hoy.""")
    
    # ========== SECCIÓN 1: INFORMACIÓN PERSONAL ==========
    with st.expander("👤 INFORMACIÓN PERSONAL", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            datos['edad'] = st.number_input(
                "Edad del estudiante",
                min_value=10,
                max_value=61,
                value=17,
                help="Edad en años"
            )
            
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
            datos['pertenece_etnia'] = st.selectbox(
                "¿Pertenece a alguna etnia?",
                options=['No', 'Si'],
                index=0
            )
            
            datos['presento_fuera_edad'] = st.selectbox(
                "¿Presentó la prueba fuera de la edad esperada?",
                options=['0', '1'],
                index=0,
                format_func=lambda x: "No" if x == "0" else "Sí"
            )
            
            datos['region'] = st.selectbox(
                "Región de residencia",
                options=['Andina', 'Caribe', 'Pacífica', 'Orinoquía', 'Amazónica'],
                index=0
            )
    
    # ========== SECCIÓN 2: VIVIENDA Y HOGAR ==========
    with st.expander("🏠 CARACTERÍSTICAS DE LA VIVIEDA"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            datos['estrato_casa'] = st.selectbox(
                "Estrato de la vivienda",
                options=['Estrato 1', 'Estrato 2', 'Estrato 3', 'Estrato 4', 
                        'Estrato 5', 'Estrato 6', 'Sin estrato'],
                index=1
            )
        
        with col2:
            datos['num_personas_casa'] = st.selectbox(
                "Número de personas en el hogar",
                options=['1 a 2', '3 a 4', '5 a 6', '7 a 8', '9 o más'],
                index=1
            )
        
        with col3:
            datos['cuartos_casa'] = st.selectbox(
                "Número de cuartos en la vivienda",
                options=['Uno', 'Dos', 'Tres', 'Cuatro', 'Cinco', 'Seis o mas'],
                index=2
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
            datos['educacion_padre'] = st.selectbox(
                "Nivel educativo del padre",
                options=['Bachiller', 'Primaria incompleta', 'Bachiller incompleta',
                         'Profesional', 'Primaria', 'Tecnico/Tecnologo', 'Ninguna',
                         'Profesional incompleta', 'Tecnico/Tecnologo incompleta', 
                         'No aplica', 'Postgrado'],
                index=0
            )
            
            datos['actividad_padre'] = st.selectbox(
                "Actividad económica del padre",
                options=['Trabajadores operativos', 'Microempresario', 'Sector primario',
                        'Trabajador independiente', 'Sin informacion',
                        'Profesionales', 'Sin actividad remunerada',
                        'Directivos', 'Pensionado'],
                index=0
            )
        
        with col2:
            datos['educacion_madre'] = st.selectbox(
                "Nivel educativo de la madre",
                options=['Bachiller', 'Primaria incompleta', 'Bachiller incompleta',
                         'Profesional', 'Tecnico/Tecnologo', 'Primaria',
                         'Tecnico/Tecnologo incompleta', 'Profesional incompleta', 
                         'Ninguna', 'No aplica', 'Postgrado'],
                index=0
            )
            
            datos['actividad_madre'] = st.selectbox(
                "Actividad económica de la madre",
                options=['Sin actividad remunerada', 'Trabajadores operativos',
                        'Microempresario', 'Profesionales', 'Trabajador independiente',
                        'Sector primario', 'Sin informacion', 'Directivos', 'Pensionado'],
                index=0
            )
        
        datos['educacion_padres'] = st.selectbox(
            "Nivel educativo combinado de los padres",
            options=['Al Menos Un Bachiller', 'Educación Superior', 'Educación Primaria Incompleta',
                    'Educación Técnica', 'Bachillerato Completo', 'Educación Primaria',
                    'Educación Secundaria Incompleta', 'Sin Información',
                    'Educación Superior Incompleta', 'No Aplica', 'Sin Educación Formal'],
            index=0
        )
    
    # ========== SECCIÓN 5: HÁBITOS Y COSTUMBRES ==========
    with st.expander("📚 HÁBITOS DE ESTUDIO Y LECTURA"):
        col1, col2 = st.columns(2)
        
        with col1:
            datos['num_libros_casa'] = st.selectbox(
                "¿Cuántos libros hay en casa?",
                options=['0 a 10 libros', '11 a 25 libros', '26 a 100 libros', 'Más de 100 libros'],
                index=0
            )
            
            datos['cuanto_lee'] = st.selectbox(
                "¿Cuánto lee por entretenimiento?",
                options=['30 minutos o menos', 'Entre 30 y 60 minutos',
                        'No leo por entretenimiento', 'Entre 1 y 2 horas', 'Más de 2 horas'],
                index=0
            )
            
            datos['perfil_lector'] = st.selectbox(
                "Perfil de lector",
                options=['Poco Apoyo, Poco Habito', 'Poco Apoyo, Buen Habito',
                        'Buen Apoyo, Buen Habito', 'Buen Apoyo, Poco Habito', 'Desconocido'],
                index=0
            )
        
        with col2:
            datos['cuanto_navega_web'] = st.selectbox(
                "¿Cuánto navega en internet?",
                options=['Entre 1 y 3 horas', 'Más de 3 horas', 'Entre 30 y 60 minutos',
                        '30 minutos o menos', 'No navega internet'],
                index=0
            )
            
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
    """Transforma los datos del formulario al formato one-hot que espera el modelo"""
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
        'educacion_padre_bachiller',
        'educacion_padre_bachiller incompleta', 
        'educacion_padre_ninguna',
        'educacion_padre_no aplica',
        'educacion_padre_postgrado',
        'educacion_padre_primaria',
        'educacion_padre_primaria incompleta',
        'educacion_padre_profesional',
        'educacion_padre_profesional incompleta',
        'educacion_padre_tecnico/tecnologo',
        'educacion_padre_tecnico/tecnologo incompleta',
        'educacion_padre_nan',
        'educacion_madre_bachiller',
        'educacion_madre_bachiller incompleta', 
        'educacion_madre_ninguna',
        'educacion_madre_no aplica', 
        'educacion_madre_postgrado',
        'educacion_madre_primaria',
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
    
    df_one_hot = pd.DataFrame(0, index=[0], columns=columnas_modelo)
    
    binarias = ['internet', 'tv', 'computador', 'lavadora', 'microndas',
               'carro', 'moto', 'consola', 'colegio_bilingue']
    
    for var in binarias:
        if var in datos_usuario:
            if isinstance(datos_usuario[var], bool):
                df_one_hot[var] = 1 if datos_usuario[var] else 0
            elif isinstance(datos_usuario[var], str):
                df_one_hot[var] = 1 if datos_usuario[var].lower() in ['si', 'sí', 'yes', 'true', '1'] else 0
    
    if 'edad' in datos_usuario:
        df_one_hot['edad'] = datos_usuario['edad']
    
    if 'presento_fuera_edad' in datos_usuario:
        df_one_hot['presento_fuera_edad'] = int(datos_usuario['presento_fuera_edad'])
    
    mapeo_categorias = {
        'estrato_casa': {
            'Estrato 1': 'estrato_casa_estrato 1',
            'Estrato 2': 'estrato_casa_estrato 2',
            'Estrato 3': 'estrato_casa_estrato 3',
            'Estrato 4': 'estrato_casa_estrato 4',
            'Estrato 5': 'estrato_casa_estrato 5',
            'Estrato 6': 'estrato_casa_estrato 6',
            'Sin estrato': 'estrato_casa_sin estrato'
        },
        'num_personas_casa': {
            '1 a 2': 'num_personas_casa_1 a 2',
            '3 a 4': 'num_personas_casa_3 a 4',
            '5 a 6': 'num_personas_casa_5 a 6',
            '7 a 8': 'num_personas_casa_7 a 8',
            '9 o más': 'num_personas_casa_9 o más'
        },
        'cuartos_casa': {
            'Uno': 'cuartos_casa_uno',
            'Dos': 'cuartos_casa_dos',
            'Tres': 'cuartos_casa_tres',
            'Cuatro': 'cuartos_casa_cuatro',
            'Cinco': 'cuartos_casa_cinco',
            'Seis o mas': 'cuartos_casa_seis o mas'
        },
        'region': {
            'Amazónica': 'region_amazónica',
            'Andina': 'region_andina',
            'Caribe': 'region_caribe',
            'Orinoquía': 'region_orinoquía',
            'Pacífica': 'region_pacífica'
        },
        'educacion_padre': {
            'Bachiller': 'educacion_padre_bachiller',
            'Bachiller incompleta': 'educacion_padre_bachiller incompleta',
            'Ninguna': 'educacion_padre_ninguna',
            'No aplica': 'educacion_padre_no aplica',
            'Postgrado': 'educacion_padre_postgrado',
            'Primaria': 'educacion_padre_primaria',
            'Primaria incompleta': 'educacion_padre_primaria incompleta',
            'Profesional': 'educacion_padre_profesional',
            'Profesional incompleta': 'educacion_padre_profesional incompleta',
            'Tecnico/Tecnologo': 'educacion_padre_tecnico/tecnologo',
            'Tecnico/Tecnologo incompleta': 'educacion_padre_tecnico/tecnologo incompleta'
        },
        'educacion_madre': {
            'Bachiller': 'educacion_madre_bachiller',
            'Bachiller incompleta': 'educacion_madre_bachiller incompleta',
            'Ninguna': 'educacion_madre_ninguna',
            'No aplica': 'educacion_madre_no aplica',
            'Postgrado': 'educacion_madre_postgrado',
            'Primaria': 'educacion_madre_primaria',
            'Primaria incompleta': 'educacion_madre_primaria incompleta',
            'Profesional': 'educacion_madre_profesional',
            'Profesional incompleta': 'educacion_madre_profesional incompleta',
            'Tecnico/Tecnologo': 'educacion_madre_tecnico/tecnologo',
            'Tecnico/Tecnologo incompleta': 'educacion_madre_tecnico/tecnologo incompleta'
        },
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
    
    for var_original, mapeo in mapeo_categorias.items():
        if var_original in datos_usuario:
            valor = datos_usuario[var_original]
            if valor in mapeo:
                columna_one_hot = mapeo[valor]
                if columna_one_hot in df_one_hot.columns:
                    df_one_hot[columna_one_hot] = 1
    
    variables_no_mapeadas = [var for var in datos_usuario.keys() 
                            if var not in binarias 
                            and var not in ['edad', 'presento_fuera_edad']
                            and var not in mapeo_categorias]
    
    for var_original in variables_no_mapeadas:
        valor = datos_usuario[var_original]
        if isinstance(valor, str):
            valor_limpio = valor.lower().replace(' ', '_').replace('/', '_').replace('ó', 'o')
            
            encontrado = False
            for columna in df_one_hot.columns:
                var_en_columna = var_original.lower().replace('_', ' ') in columna.lower()
                valor_en_columna = valor_limpio in columna.lower().replace(' ', '_')
                
                if var_en_columna and valor_en_columna:
                    df_one_hot[columna] = 1
                    encontrado = True
                    break
            
            if not encontrado:
                for columna in df_one_hot.columns:
                    if var_original in columna:
                        palabras_valor = valor.lower().split()
                        for palabra in palabras_valor:
                            if palabra in columna.lower() and len(palabra) > 2:
                                df_one_hot[columna] = 1
                                encontrado = True
                                break
                    if encontrado:
                        break
    
    return df_one_hot

def crear_grafico_distribucion(prediccion_usuario, ma_error=MAE_TEST, promedio_icfes=PROMEDIO_ICFES):
    """Crea gráfico de distribución con predicción del usuario y rango de error - CORREGIDO"""
    
    media = promedio_icfes
    desviacion = DESV_STD_ICFES
    
    # Generar puntos para la curva normal
    x = np.linspace(max(0, media - 4*desviacion), min(500, media + 4*desviacion), 1000)
    y = (1/(desviacion*np.sqrt(2*np.pi))) * np.exp(-0.5*((x-media)/desviacion)**2)
    
    # Calcular percentiles
    percentiles = {
        '25%': np.percentile(np.random.normal(media, desviacion, 10000), 25),
        '50%': media,
        '75%': np.percentile(np.random.normal(media, desviacion, 10000), 75)
    }
    
    # Crear figura con Plotly
    fig = go.Figure()
    
    # 1. Área de distribución con gradiente
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill='tozeroy',
        fillcolor=f'rgba(74, 144, 226, 0.25)',
        line=dict(color=COLORES['primario'], width=3),
        name='Distribución ICFES',
        hovertemplate='<b>Puntaje:</b> %{x:.0f} pts<br><b>Densidad:</b> %{y:.4f}<extra></extra>'
    ))
    
    # 2. Línea del promedio nacional - POSICIÓN FIJADA
    fig.add_vline(
        x=promedio_icfes,
        line_dash="dash",
        line_color=COLORES['advertencia'],
        line_width=3,
        annotation=dict(
            text=f"<b>Promedio Nacional</b><br>{promedio_icfes:.0f} pts",
            font=dict(size=13, color=COLORES['advertencia'], family="Arial Black"),
            bgcolor="rgba(255, 179, 71, 0.15)",
            bordercolor=COLORES['advertencia'],
            borderwidth=2,
            x=0.02,  # Posición izquierda fija (2% desde la izquierda)
            y=0.95,  # Posición superior (95% desde abajo)
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor='left'
        )
    )
    
    # 3. Línea de tu predicción - POSICIÓN FIJADA
    color_prediccion = COLORES['exito'] if prediccion_usuario > promedio_icfes else COLORES['info']
    fig.add_vline(
        x=prediccion_usuario,
        line_dash="solid",
        line_color=color_prediccion,
        line_width=4,
        annotation=dict(
            text=f"<b>Tu Predicción</b><br>{prediccion_usuario:.0f} pts",
            font=dict(size=14, color=color_prediccion, family="Arial Black"),
            bgcolor=f"rgba({int(color_prediccion[1:3], 16)}, {int(color_prediccion[3:5], 16)}, {int(color_prediccion[5:7], 16)}, 0.2)",
            bordercolor=color_prediccion,
            borderwidth=3,
            x=0.02,
            y=0.75,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor='left'
        )
    )
    
    # 4. Área sombreada del rango de error (Rango de Confianza)
    fig.add_vrect(
        x0=max(0, prediccion_usuario - ma_error),
        x1=min(500, prediccion_usuario + ma_error),
        fillcolor="rgba(123, 104, 238, 0.15)",
        line_width=0,
        layer="below",
        annotation=dict(
            text=f"Rango de confianza<br>±{ma_error:.0f} pts",
            font=dict(size=14, color='#000000'),
            showarrow=False
        )
    )
    
    # 5. Límites del rango de error (sin anotaciones)
    fig.add_vline(
        x=max(0, prediccion_usuario - ma_error),
        line_dash="dot",
        line_color=COLORES['secundario'],
        line_width=2,
        opacity=0.6
    )
    
    fig.add_vline(
        x=min(500, prediccion_usuario + ma_error),
        line_dash="dot",
        line_color=COLORES['secundario'],
        line_width=2,
        opacity=0.6
    )
    
    # 6. Percentiles con estilo mejorado
    for i, (label, value) in enumerate(percentiles.items()):
        fig.add_vline(
            x=value,
            line_dash="dot",
            line_color="rgba(150, 150, 150, 0.4)",
            line_width=1
        )
    
    # Configurar diseño - COLORES DE EJES CORREGIDOS
    fig.update_layout(
        title=dict(
            text="Tu predicción en el contexto nacional",
            font=dict(size=24, color=COLORES['texto_oscuro'], family="Arial Black"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="<b>Puntaje ICFES</b>",
            title_font=dict(size=14, color='#000000'),  # NEGRO
            tickfont=dict(size=12, color='#000000'),    # NEGRO
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=1,
            range=[max(0, media - 3*desviacion), min(500, media + 3*desviacion)],
            zeroline=True,
            zerolinecolor='rgba(150, 150, 150, 0.5)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title="<b>Densidad de Probabilidad</b>",
            title_font=dict(size=14, color='#000000'),  # NEGRO
            tickfont=dict(size=12, color='#000000'),    # NEGRO
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            gridwidth=1,
            zeroline=True,
            zerolinecolor='rgba(150, 150, 150, 0.5)',
            zerolinewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',  # Fondo blanco para mejor contraste
        showlegend=False,
        hovermode="x unified",
        height=550,
        margin=dict(l=80, r=40, t=100, b=80)  # Más margen izquierdo para las anotaciones
    )
    
    # Agregar anotaciones de percentiles manualmente
    for i, (label, value) in enumerate(percentiles.items()):
        fig.add_annotation(
            x=value,
            y=0.02,
            text=label,
            font=dict(size=10, color='gray'),
            showarrow=False,
            xref="x",
            yref="paper",
            yanchor='bottom'
        )
    
    return fig

def mostrar_animacion_globos():
    """Muestra animación de globos cuando el puntaje es bueno"""
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <div class="balloon">🎈</div>
        <div class="balloon" style="animation-delay: 0.5s">🎉</div>
        <div class="balloon" style="animation-delay: 1s">🎊</div>
        <div class="balloon" style="animation-delay: 1.5s">⭐</div>
        <div class="balloon" style="animation-delay: 2s">🎈</div>
    </div>
    """, unsafe_allow_html=True)

def mostrar_mensaje_resultado(prediccion, promedio=PROMEDIO_ICFES, ma_error=MAE_TEST):
    """Muestra mensaje personalizado basado en el resultado"""
    
    diferencia = prediccion - promedio
    
    if diferencia > ma_error * 2:
        return "🎉 ¡EXCELENTE! Tu puntaje predicho está MUY por encima del promedio.", "success"
    elif diferencia > ma_error:
        return "👍 ¡BUEN TRABAJO! Tu puntaje predicho está por encima del promedio.", "success"
    elif abs(diferencia) <= ma_error:
        return "🤔 Tu puntaje predicho es similar al promedio nacional.", "info"
    elif diferencia < -ma_error:
        return "📚 Hay oportunidades para mejorar. Considera reforzar tu preparación.", "warning"
    else:
        return "🔍 Se recomienda atención especial. Considera buscar apoyo adicional.", "warning"

def crear_grafico_errores_mejorado():
    """Crea gráfico de distribución de errores con colores mejorados"""
    errores_simulados = np.random.normal(0, MAE_TEST/0.7979, 10000)
    
    fig = go.Figure()
    
    # Histograma con colores mejorados
    fig.add_trace(go.Histogram(
        x=errores_simulados,
        nbinsx=60,
        name='Errores del modelo',
        marker=dict(
            color=errores_simulados,
            colorscale=[
                [0, COLORES['error']],
                [0.5, COLORES['info']],
                [1, COLORES['exito']]
            ],
            line=dict(color='white', width=1)
        ),
        opacity=0.8
    ))
    
    # Líneas de MAE con anotaciones fijas
    fig.add_vline(
        x=MAE_TEST,
        line_dash="dash",
        line_color=COLORES['advertencia'],
        line_width=3,
        annotation=dict(
            text=f"<b>+MAE</b><br>{MAE_TEST:.1f} pts",
            font=dict(size=12, color=COLORES['advertencia']),
            bgcolor="rgba(255, 179, 71, 0.15)",
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor='left'
        )
    )
    fig.add_vline(
        x=-MAE_TEST,
        line_dash="dash",
        line_color=COLORES['advertencia'],
        line_width=3,
        annotation=dict(
            text=f"<b>-MAE</b><br>{-MAE_TEST:.1f} pts",
            font=dict(size=12, color=COLORES['advertencia']),
            bgcolor="rgba(255, 179, 71, 0.15)",
            x=0.05,
            y=0.85,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor='left'
        )
    )
    
    # Línea del cero
    fig.add_vline(
        x=0,
        line_dash="solid",
        line_color=COLORES['exito'],
        line_width=2,
        annotation=dict(
            text="<b>Error = 0</b>",
            font=dict(size=12, color=COLORES['exito']),
            x=0.05,
            y=0.75,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor='left'
        )
    )
    
    # Configurar diseño con colores de ejes en negro
    fig.update_layout(
        title=dict(
            text="📉 Distribución de Errores del Modelo",
            font=dict(size=20, color=COLORES['texto_oscuro'], family="Arial Black"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="<b>Error de Predicción (puntos)</b>",
            title_font=dict(size=13, color='#000000'),  # NEGRO
            tickfont=dict(size=11, color='#000000'),    # NEGRO
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=True,
            zerolinecolor='rgba(150, 150, 150, 0.5)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title="<b>Frecuencia</b>",
            title_font=dict(size=13, color='#000000'),  # NEGRO
            tickfont=dict(size=11, color='#000000'),    # NEGRO
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=True,
            zerolinecolor='rgba(150, 150, 150, 0.5)',
            zerolinewidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        height=450,
        margin=dict(l=80, r=40, t=80, b=80)
    )
    
    return fig

def main():
    """Función principal de la aplicación"""

    modelo = cargar_modelo()

    if modelo is None:
        st.error("No se pudo cargar el modelo. Verifica la ruta del archivo.")
        st.stop()

    st.sidebar.title("🔍 Navegación")
    pagina = st.sidebar.radio(
        "Selecciona una página:",
        ["🏠 Inicio", "🎯 Predicción", "📊 Análisis"]
    )

    if pagina == "🏠 Inicio":
        st.title("Bienvenido al Predictor de Puntajes ICFES")
        st.write(
            "Esta herramienta utiliza un modelo de Random Forest para predecir los puntajes del ICFES "
            "a partir de variables socioeconómicas."
        )

        st.divider()

        st.markdown(
            """
### **Instrucciones:**
1. Navega a la pestaña **Predicción**
2. Completa el formulario con la información requerida
3. Haz clic en **Predecir Puntaje**
4. Revisa los resultados y análisis"""
        )
        
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("¿Qué es este proyecto?")
            st.write(
                "Este modelo hace parte de un **proyecto personal** cuyo objetivo fue recorrer "
                "todo el ciclo de vida de un sistema de Machine Learning: preparación de datos, "
                "entrenamiento, evaluación y despliegue en una aplicación interactiva."
            )
            st.write(
                "El modelo utilizado es un **Random Forest**, entrenado con datos de resultados "
                "del ICFES del año **2020**."
            )

        with col2:
            st.subheader("⚠️ Limitaciones importantes")
            st.warning(
                "El modelo fue entrenado únicamente con variables socioeconómicas, "
                "las cuales no explican completamente el desempeño en el examen."
            )
            st.write(
                "Por esta razón, el modelo alcanza un **R² cercano al 35%**, lo que indica que "
                "las predicciones deben interpretarse como una **aproximación exploratoria**, "
                "no como un resultado exacto."
            )

        st.divider()

        # MOTIVACIÓN
        st.subheader("Motivación del proyecto")
        st.write(
            "Existe una percepción común de que los exámenes actuales son “más fáciles” que en el pasado. "
            "Este proyecto permite que personas que presentaron el ICFES hace años puedan estimar "
            "cómo les iría si lo presentaran hoy."
        )
        st.write(
            "Aunque los datos corresponden a 2020, la **metodología del examen no ha cambiado "
            "de forma significativa**, por lo que el ejercicio sigue siendo válido como análisis."
        )

        with st.expander("📊 Métricas del Modelo"):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "MAE (Test)", f"{MAE_TEST:.1f} pts",
                    "Error absoluto medio"
                )

            with col2:
                st.metric(
                    "Promedio Nacional", f"{PROMEDIO_ICFES:.0f} pts",
                    "Puntaje promedio ICFES"
                )

            with col3:
                st.metric(
                    "Desviación Estándar", f"{DESV_STD_ICFES:.1f} pts",
                    "Variabilidad de puntajes"
                )

    elif pagina == "🎯 Predicción":
        datos_usuario = crear_formulario()

        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(
                "🔮 Predecir Puntaje ICFES",
                type="primary",
                use_container_width=True
            ):
                with st.spinner("Procesando datos y calculando predicción..."):
                    try:
                        datos_one_hot = transformar_a_one_hot(datos_usuario)
                        prediccion = modelo.predict(datos_one_hot)
                        puntaje_predicho = float(prediccion[0])

                        st.success("✅ Predicción completada")
                        st.markdown("---")

                        with st.container():
                            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
                            with col_res2:
                                st.markdown(
                                    f"""
<div style="text-align: center; padding: 30px; border-radius: 15px; 
            background: linear-gradient(135deg, {COLORES['degradado_1']} 0%, {COLORES['degradado_2']} 100%);
            color: white; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
    <h2 style="margin: 0; font-size: 1.8rem; font-weight: 600;">Puntaje Predicho</h2>
    <h1 style="margin: 15px 0; font-size: 5rem; font-weight: 800;">{puntaje_predicho:.0f}</h1>
    <p style="margin: 0; font-size: 1.3rem; opacity: 0.95;">puntos ICFES</p>
</div>
""",
                                    unsafe_allow_html=True
                                )

                        st.markdown("---")
                        st.markdown("### 📊 Interpretación del Resultado")

                        col_met1, col_met2, col_met3 = st.columns(3)

                        with col_met1:
                            st.metric(
                                label="Puntaje Predicho",
                                value=f"{puntaje_predicho:.0f}",
                                delta=f"{puntaje_predicho - PROMEDIO_ICFES:+.0f} vs promedio",
                                delta_color="normal"
                            )

                        with col_met2:
                            st.metric(
                                label="Margen de Error",
                                value=f"±{MAE_TEST:.1f} pts",
                                help="Error absoluto medio del modelo en datos de prueba"
                            )

                        with col_met3:
                            rango_inferior = max(0, puntaje_predicho - MAE_TEST)
                            rango_superior = min(500, puntaje_predicho + MAE_TEST)
                            st.metric(
                                label="Rango Probable",
                                value=f"{rango_inferior:.0f}-{rango_superior:.0f}",
                                help="Tu puntaje real probablemente esté en este rango"
                            )

                        mensaje, tipo = mostrar_mensaje_resultado(puntaje_predicho)

                        if tipo == "success":
                            st.success(mensaje)
                            if puntaje_predicho > PROMEDIO_ICFES:
                                st.markdown("---")
                                mostrar_animacion_globos()
                        elif tipo == "warning":
                            st.warning(mensaje)
                        else:
                            st.info(mensaje)

                        st.markdown("---")
                        st.markdown("### 📈 Visualización de la Predicción")

                        fig = crear_grafico_distribucion(puntaje_predicho)
                        st.plotly_chart(fig, use_container_width=True)

                        with st.expander("📖 ¿Cómo interpretar este gráfico?"):
                            st.write(
                                f"""
**Elementos clave del gráfico:**

1. **Curva azul**: Distribución típica de puntajes ICFES
2. **Línea naranja (---)**: Puntaje promedio nacional ({PROMEDIO_ICFES:.0f} pts)
3. **Línea verde/azul sólida**: Tu puntaje predicho ({puntaje_predicho:.0f} pts)
4. **Área púrpura**: Rango de confianza (±{MAE_TEST:.1f} puntos)

**Interpretación importante:**
- El modelo tiene un **error medio de {MAE_TEST:.1f} puntos**
- Tu puntaje real probablemente esté entre **{max(0, puntaje_predicho - MAE_TEST):.0f} y {min(500, puntaje_predicho + MAE_TEST):.0f} puntos**
- La predicción es una **estimación**, no un valor exacto
"""
                            )

                        st.markdown("---")
                        st.markdown("### 🔍 Factores que Influyeron en la Predicción")

                        with st.expander("Ver detalles técnicos"):
                            st.write(
                                f"""
**Métricas del modelo:**
- **MAE (Test):** {MAE_TEST:.1f} puntos
- **Promedio histórico:** {PROMEDIO_ICFES:.0f} puntos
- **Desviación estándar:** {DESV_STD_ICFES:.1f} puntos

**Interpretación del margen de error:**
1. El modelo se equivoca en promedio por ±{MAE_TEST:.1f} puntos
2. Esto representa el {MAE_TEST / DESV_STD_ICFES * 100:.1f}% de la variabilidad total
3. Tu puntaje real podría variar hasta {MAE_TEST:.0f} puntos
"""
                            )

                        if puntaje_predicho < PROMEDIO_ICFES - MAE_TEST:
                            with st.expander("💡 Recomendaciones para mejorar"):
                                st.write(
                                    """
**Acciones sugeridas:**

1. **Refuerzo académico**: Enfócate en áreas con menor desempeño
2. **Simulacros**: Realiza pruebas prácticas regularmente
3. **Plan de estudio**: 2-3 horas diarias de estudio constante
4. **Recursos**: Utiliza guías oficiales del ICFES
5. **Apoyo**: Considera tutorías o preicfes gratuitos
6. **Hábitos**: Duerme bien y aliméntate adecuadamente
"""
                                )

                        with st.expander("🔍 Ver datos procesados"):
                            st.write("**Datos en formato one-hot:**")
                            st.dataframe(datos_one_hot.T[datos_one_hot.T[0] > 0])

                    except Exception as e:
                        st.error(f"❌ Error al hacer la predicción: {str(e)}")

    elif pagina == "📊 Análisis":
        st.title("📊 Análisis del Modelo")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MAE en Test", f"{MAE_TEST:.1f} pts", "Error absoluto medio")
            st.metric("RMSE en Test", "38.9 pts", "Raíz del error cuadrático medio")

        with col2:
            st.metric("R² en Test", "0.3566", "Varianza explicada")
            st.metric("MedAE en Test", "27.1 pts", "Mediana del error absoluto")

        st.markdown("---")
        st.subheader("📈 Distribución del Error del Modelo")

        fig_errores = crear_grafico_errores_mejorado()
        st.plotly_chart(fig_errores, use_container_width=True)

        with st.expander("📖 Interpretación del análisis"):
            st.write(
                f"""
**Análisis de las métricas:**

1. **MAE = {MAE_TEST:.1f} puntos**: El modelo se equivoca en promedio por ±{MAE_TEST:.1f} puntos
2. **R² = 0.3566**: El modelo explica el 35.7% de la variabilidad en los puntajes
3. **Comparación**: Nuestro modelo reduce el error en un {((DESV_STD_ICFES * 0.8 - MAE_TEST) / (DESV_STD_ICFES * 0.8) * 100):.1f}% respecto a un modelo base

**Limitaciones:**
- Solo considera variables socioeconómicas
- No incluye hábitos de estudio individuales
- Factores no observables representan gran parte de la variabilidad
"""
            )


if __name__ == "__main__":
    main()
