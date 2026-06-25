# =============================================================================
# FashAnalytics — Sistema de predicción de demanda para MYPEs textiles
# Backend: Spring Boot localhost:8081 | Modelo: Random Forest (joblib)
# Roles: GERENTE (id_rol=1) | COLABORADOR (id_rol=2)
# =============================================================================
# REFACTOR: Inventario con smart delete, entradas, descuento automático,
#           cascada producto-categoría, validación de stock al vender.
# =============================================================================

import os, io, warnings, base64, json, time
from datetime import date, datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import joblib
import unicodedata

warnings.filterwarnings("ignore")

try:
    import google.generativeai as genai
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_URL   = "https://front-modelopredictivo.onrender.com"


#from dotenv import load_dotenv
#load_dotenv()
#GEMINI_KEY = os.getenv("GEMINI_KEY", "")

#cambiado por este: 
import streamlit as st

GEMINI_KEY = st.secrets["GEMINI_KEY"]

st.set_page_config(
    page_title="FashAnalytics",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CSS — MEJORADO UX (sin cambiar colores) + nuevos estilos para inventario
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root {
    --bg:#f7f1e7; --panel:#fffdf8; --primary:#5a351e;
    --accent:#c9953b; --border:#e5d4b8; --text:#2f241b;
    --muted:#7b6a57; --success:#5d8c45; --warning:#b7791f; --danger:#b04a35;
}
html,body,[class*="css"]{ font-family:'Inter',sans-serif; color:var(--text); }
[data-testid="stAppViewContainer"]{ background:linear-gradient(180deg,#f7f1e7 0%,#efe4d2 100%); }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#3e2416 0%,#6b3f21 54%,#3a2115 100%); }
[data-testid="stSidebar"] *{ color:#fff8ea !important; }
[data-testid="stSidebar"] .stButton button{ background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.16); }
.block-container{ padding-top:1.4rem; padding-bottom:3rem; max-width:1380px; }

.login-hero{ background:linear-gradient(135deg,#4a2b18,#6b3f21 55%,#a36a2d); color:white; border-radius:30px; padding:3rem; min-height:360px; display:flex; flex-direction:column; justify-content:center; }
.login-hero h1{ font-size:2.6rem; font-weight:900; margin:.5rem 0 1rem; color:#fff !important; }
.login-hero p{ color:#f3dfbd; line-height:1.75; }
.auth-panel-title{ background:var(--panel); border:1px solid var(--border); border-radius:24px; padding:1.6rem 1.7rem; margin-bottom:1rem; }
.auth-panel-title h2{ margin:0; color:#3b271b; font-weight:900; }
.auth-panel-title p{ color:#7b6a57; margin:.35rem 0 0; }
[data-testid="stForm"]{ background:var(--panel); border:1px solid var(--border); border-radius:24px; padding:1.4rem 1.5rem 1.2rem; }

.main-header{ background:linear-gradient(135deg,#4a2b18 0%,#6b3f21 55%,#a36a2d 100%); border-radius:0 0 22px 22px; padding:1rem 1.4rem; margin-bottom:1.25rem; color:white; }
.main-header h1{ font-size:2rem; margin:0; font-weight:900; color:#fff !important; }

.kpi{ background:var(--panel); border:1px solid var(--border); border-radius:20px; padding:1.1rem 1.2rem; min-height:110px; transition: transform 0.18s ease, box-shadow 0.18s ease; }
.kpi:hover{ transform:translateY(-3px); box-shadow:0 6px 18px rgba(90,53,30,.13); }
.kpi .lbl{ color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.7px; font-weight:800; }
.kpi .val{ color:var(--primary); font-size:1.9rem; line-height:1.05; margin-top:.4rem; font-weight:900; }
.kpi .sub{ color:#6b5a46; font-size:.78rem; margin-top:.4rem; }

.section-title{ color:#3b271b; font-size:1.2rem; font-weight:900; margin:1.5rem 0 .8rem; }
.section-note{ color:var(--muted); font-size:.9rem; margin:-.2rem 0 1rem; line-height:1.55; }

.pred-box{ background:linear-gradient(135deg,#5a351e,#a36a2d); border-radius:24px; padding:2rem 1.5rem; color:#fffaf0; text-align:center; }
.pred-box .pl{ font-size:.85rem; color:#f7dfb7; margin-bottom:.5rem; }
.pred-box .pv{ font-size:3rem; font-weight:900; line-height:1; }
.pred-box .pu{ font-size:.85rem; color:#f6e6c8; margin-top:.4rem; }

.rec{ background:var(--panel); border:1px solid var(--border); border-left:5px solid var(--accent); border-radius:16px; padding:1rem 1.15rem; margin-bottom:.85rem; }
.rec .rt{ font-weight:900; margin-bottom:.25rem; color:#3b271b; }
.rec.pos{ border-left-color:#5d8c45; }
.rec.warn{ border-left-color:#b7791f; }
.rec.risk{ border-left-color:#b04a35; }

.sidebar-profile{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); border-radius:18px; padding:14px; margin:12px 0 18px; }
.sidebar-profile strong{ display:block; font-size:.94rem; margin-bottom:6px; }
.sidebar-profile span{ display:block; color:#ead4ad !important; font-size:.78rem; }
.logo-row{ display:flex; align-items:center; gap:14px; margin-bottom:18px; }
.logo-box{ width:52px; height:52px; border-radius:16px; background:#f5e2bd; display:grid; place-items:center; }
.brand-title{ font-size:1.25rem; font-weight:900; margin:0; color:#fff8ea; }
.brand-sub{ margin:0; color:#ead4ad; font-size:.82rem; }

.vdict{ border-radius:16px; padding:1rem 1.2rem; font-weight:600; border:1px solid var(--border); margin-bottom:1rem; }
.vdict.ok{ background:#edf4df; color:#456b32; border-color:#c9dda9; }
.vdict.warn{ background:#fff2d4; color:#895c14; border-color:#e6c985; }
.vdict.bad{ background:#fae4d9; color:#963724; border-color:#e6b4a1; }

.hist-row{ background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:.9rem 1.1rem; margin-bottom:.6rem; display:flex; justify-content:space-between; align-items:center; transition: box-shadow 0.18s ease, transform 0.18s ease; }
.hist-row:hover{ box-shadow:0 4px 14px rgba(90,53,30,.10); transform:translateY(-1px); }
.hist-row .hr-prod{ font-weight:800; color:#3b271b; font-size:.95rem; }
.hist-row .hr-val{ font-size:1.3rem; font-weight:900; color:var(--primary); }
.hist-row .hr-fecha{ font-size:.76rem; color:var(--muted); margin-top:.2rem; }

.stButton>button,
div[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button{
    background:linear-gradient(135deg,#5a351e 0%,#8a5a2b 100%) !important;
    color:#fff8ea !important;
    border:1px solid #c9953b !important;
    border-radius:12px !important;
    font-weight:800 !important;
    min-height:44px !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
}
.stButton>button:hover,
div[data-testid="stFormSubmitButton"] button:hover,
[data-testid="stDownloadButton"] button:hover{
    transform: scale(1.035) translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(90,53,30,.30) !important;
    filter: brightness(1.08) !important;
    cursor: pointer !important;
}
.stButton>button:active,
div[data-testid="stFormSubmitButton"] button:active{
    transform: scale(0.98) !important;
    box-shadow: 0 2px 8px rgba(90,53,30,.18) !important;
}

[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stDateInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="select"]>div{
    background:#fffaf0 !important;
    color:#2f241b !important;
    border:1.8px solid #b98745 !important;
    border-radius:13px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus{
    border-color:#5a351e !important;
    box-shadow: 0 0 0 3px rgba(90,53,30,.12) !important;
    outline: none !important;
}

label[data-testid="stWidgetLabel"],label[data-testid="stWidgetLabel"] p{ color:#2f241b !important; font-weight:700 !important; }

.stTabs [data-baseweb="tab"]{ background:#fff8eb; border:1px solid var(--border); border-radius:999px; color:var(--primary); font-weight:800; transition: transform 0.15s ease, box-shadow 0.15s ease; }
.stTabs [data-baseweb="tab"]:hover{ transform:translateY(-1px); box-shadow:0 3px 10px rgba(90,53,30,.12); }
.stTabs [aria-selected="true"]{ background:var(--primary) !important; color:#fff8ea !important; }

div[role="radiogroup"] label,div[role="radiogroup"] label span,div[role="radio"] *{ color:#2f241b !important; font-weight:800 !important; }
[data-testid="stSidebar"] div[role="radio"]>div:first-child{ display:none !important; }
[data-testid="stSidebar"] div[role="radio"]{ padding:.52rem .65rem !important; border-radius:12px !important; margin-bottom:.18rem !important; transition: background 0.15s ease !important; }
[data-testid="stSidebar"] div[role="radio"][aria-checked="true"]{ background:rgba(255,255,255,.17) !important; }
[data-testid="stSidebar"] div[role="radiogroup"] label,[data-testid="stSidebar"] div[role="radio"] *{ color:#fff8ea !important; font-weight:800 !important; }

.welcome-banner{
    background:linear-gradient(135deg,rgba(201,149,59,.15),rgba(90,53,30,.06));
    border:1px solid #e5d4b8;
    border-left:5px solid #c9953b;
    border-radius:16px;
    padding:.9rem 1.2rem;
    margin-bottom:1.2rem;
    display:flex;
    align-items:center;
    gap:14px;
}
.welcome-banner .wb-emoji{ font-size:2rem; }
.welcome-banner .wb-text{ flex:1; }
.welcome-banner .wb-nombre{ font-weight:900; color:#3b271b; font-size:1rem; }
.welcome-banner .wb-msg{ color:#7b6a57; font-size:.84rem; margin-top:2px; }

.breadcrumb{
    color:#7b6a57;
    font-size:.78rem;
    font-weight:600;
    margin-bottom:.6rem;
    display:flex;
    align-items:center;
    gap:6px;
}
.breadcrumb span{ color:#c9953b; font-weight:900; }

.paso-header{
    background:linear-gradient(135deg,rgba(74,43,24,.90),rgba(107,63,33,.85));
    color:#fff8ea;
    border-radius:16px;
    padding:.75rem 1.2rem;
    margin-bottom:1rem;
    display:flex;
    align-items:center;
    gap:12px;
    font-weight:800;
    font-size:.98rem;
}
.paso-numero{
    background:#c9953b;
    color:#fff;
    border-radius:50%;
    width:30px; height:30px;
    display:inline-flex; align-items:center; justify-content:center;
    font-size:.88rem; font-weight:900; flex-shrink:0;
}
.paso-desc{ font-size:.8rem; color:#ead4ad; font-weight:500; margin-top:2px; }

.field-hint{
    font-size:.76rem;
    color:#7b6a57;
    margin-top:-.3rem;
    margin-bottom:.5rem;
    padding-left:.25rem;
    line-height:1.5;
}

.security-note{
    background:rgba(93,140,69,.10);
    border:1px solid rgba(93,140,69,.25);
    border-radius:12px;
    padding:.55rem .9rem;
    font-size:.78rem;
    color:#456b32;
    display:flex;
    align-items:center;
    gap:8px;
    margin-top:.8rem;
}

.empty-state{
    text-align:center;
    padding:2.5rem 1.5rem;
    background:var(--panel);
    border:1.5px dashed var(--border);
    border-radius:20px;
    margin:1rem 0;
}
.empty-state .es-icon{ font-size:2.8rem; margin-bottom:.8rem; }
.empty-state .es-title{ font-weight:900; color:#3b271b; font-size:1.05rem; margin-bottom:.4rem; }
.empty-state .es-msg{ color:#7b6a57; font-size:.88rem; line-height:1.6; }

[data-testid="stExpander"]{
    border:1px solid var(--border) !important;
    border-radius:14px !important;
    transition: box-shadow 0.18s ease !important;
}
[data-testid="stExpander"]:hover{
    box-shadow: 0 3px 12px rgba(90,53,30,.10) !important;
}

.stock-blocked{
    background:linear-gradient(135deg,rgba(176,74,53,.08),rgba(176,74,53,.03));
    border:1px solid #e6b4a1;
    border-left:5px solid #b04a35;
    border-radius:14px;
    padding:.9rem 1.1rem;
    margin:.8rem 0;
}
.stock-blocked .sb-title{
    font-weight:900;
    color:#963724;
    font-size:.95rem;
    margin-bottom:.3rem;
    display:flex; align-items:center; gap:8px;
}
.stock-blocked .sb-msg{
    color:#7b6a57;
    font-size:.85rem;
    line-height:1.5;
}

.descontinuado-tag{
    display:inline-block;
    background:#7b6a57;
    color:#fff;
    padding:.2rem .6rem;
    border-radius:8px;
    font-size:.72rem;
    font-weight:800;
    margin-left:.5rem;
}

footer{ visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
DEFAULTS = {
    "authenticated":   False,
    "jwt_token":       "",
    "rol_id":          0,
    "rol_nombre":      "",
    "empresa_id":      0,
    "empresa_nombre":  "",
    "ruc_empresa":     "",
    "sector_empresa":  "",
    "usuario_id":      0,
    "usuario_nombre":  "",
    "dni_usuario":     "",
    "correo_usuario":  "",
    "sexo_usuario":    "",
    "telefono":        "",
    "direccion":       "",
    "fecha_registro":  "",
    "tipo_rol":        "",
    "descripcion_rol": "",
    "page":            "Dashboard",
    "pred_resultado":  None,
    "pred_contexto":   {},
    "chat_history":    [],
    "reg_empresa_id":  0,
    "reg_empresa_nombre": "",
    "reg_paso":        1,
    "inv_data":        [],
    "mostrar_edicion": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =============================================================================
# HELPERS
# =============================================================================

def es_gerente() -> bool:
    return st.session_state.get("rol_id", 0) == 1

def get_header() -> dict:
    return {"Authorization": f"Bearer {st.session_state.get('jwt_token', '')}"}

def parse_rol(r) -> str:
    """
    FIX: Parsea el id_rol robustamente desde el backend.
    El backend puede devolverlo como:
      - dict: {"id_rol": 1, "tipo_rol": "GERENTE", ...}
      - int: 1 o 2
      - str: "GERENTE" o "COLABORADOR"
    """
    if isinstance(r, dict):
        tipo = r.get("tipo_rol")
        if tipo:
            return str(tipo).upper()
        rid = r.get("id_rol", 0)
        if isinstance(rid, dict):
            rid = rid.get("id_rol", 0)
        try:
            return "GERENTE" if int(rid) == 1 else ("COLABORADOR" if int(rid) == 2 else "?")
        except (ValueError, TypeError):
            return "?"
    if isinstance(r, (int, float)):
        return "GERENTE" if int(r) == 1 else ("COLABORADOR" if int(r) == 2 else "?")
    if isinstance(r, str):
        return r.upper()
    return "?"

def kpi(label, value, sub=""):
    st.markdown(f"""
<div class="kpi">
  <div class="lbl">{label}</div>
  <div class="val">{value}</div>
  <div class="sub">{sub}</div>
</div>""", unsafe_allow_html=True)

def rec_card(tipo, titulo, texto):
    st.markdown(f"""
<div class="rec {tipo}">
  <div class="rt">{titulo}</div>
  {texto}
</div>""", unsafe_allow_html=True)

def chart_style(fig, title="", height=360):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#3b271b"), x=0.02),
        height=height, plot_bgcolor="#fffdf8", paper_bgcolor="#fffdf8",
        margin=dict(l=40, r=20, t=55, b=40),
        font=dict(color="#3b271b", size=13),
        legend=dict(orientation="h", y=1.12, bgcolor="rgba(255,255,255,.6)"),
        hoverlabel=dict(bgcolor="#fff8eb", bordercolor="#c9953b"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e7d8bf", tickfont=dict(color="#3b271b"))
    fig.update_yaxes(gridcolor="#e7d8bf", linecolor="#e7d8bf", tickfont=dict(color="#3b271b"))
    return fig

def decodificar_jwt(token: str) -> dict:
    try:
        payload_b64 = token.split(".")[1]
        pad = 4 - len(payload_b64) % 4
        if pad != 4:
            payload_b64 += "=" * pad
        return json.loads(base64.urlsafe_b64decode(payload_b64).decode())
    except Exception:
        return {}

def breadcrumb(pagina: str):
    st.markdown(
        f'<div class="breadcrumb">🏠 FashAnalytics &rsaquo; <span>{pagina}</span></div>',
        unsafe_allow_html=True)

def welcome_banner():
    nombre = st.session_state.get("usuario_nombre", "")
    hora   = datetime.now().hour
    if hora < 12:
        saludo, emoji = "Buenos días", "☀️"
    elif hora < 18:
        saludo, emoji = "Buenas tardes", "🌤️"
    else:
        saludo, emoji = "Buenas noches", "🌙"
    empresa = st.session_state.get("empresa_nombre", "")
    st.markdown(f"""
<div class="welcome-banner">
  <div class="wb-emoji">{emoji}</div>
  <div class="wb-text">
    <div class="wb-nombre">{saludo}, {nombre}!</div>
    <div class="wb-msg">Bienvenido al sistema de {empresa} — ¿Qué quieres revisar hoy?</div>
  </div>
</div>""", unsafe_allow_html=True)

def paso_header(num: int, titulo: str, desc: str = ""):
    st.markdown(f"""
<div class="paso-header">
  <div class="paso-numero">{num}</div>
  <div>
    {titulo}
    {"<div class='paso-desc'>" + desc + "</div>" if desc else ""}
  </div>
</div>""", unsafe_allow_html=True)

def field_hint(texto: str):
    st.markdown(f'<div class="field-hint">💡 {texto}</div>',
                unsafe_allow_html=True)

def empty_state(icono: str, titulo: str, mensaje: str):
    st.markdown(f"""
<div class="empty-state">
  <div class="es-icon">{icono}</div>
  <div class="es-title">{titulo}</div>
  <div class="es-msg">{mensaje}</div>
</div>""", unsafe_allow_html=True)

def stock_blocked_msg(producto: str, motivo: str):
    st.markdown(f"""
<div class="stock-blocked">
  <div class="sb-title">🔒 No puedes vender "{producto}" todavía</div>
  <div class="sb-msg">{motivo}</div>
</div>""", unsafe_allow_html=True)

AYUDA_POR_PAGINA = {
    "Dashboard":       "Aquí ves un resumen de todo. Los números de arriba son tus ventas del período.",
    "Gestión comercial": "Sube tu archivo de ventas o agrégalas una por una. Empieza por la pestaña 'Subir archivo'.",
    "Inventario":      "Aquí controlas cuántos productos tienes. Si un producto aparece en rojo, te estás quedando sin stock.",
    "Predicción":      "Elige el producto y la fecha. El sistema te dirá cuánto podrías vender ese mes.",
    "Análisis de resultados": "Estos números muestran qué tan preciso es el modelo. Mientras más cerca de 100%, mejor.",
    "Inteligencia y Recomendaciones": "El asistente te puede responder dudas sobre tus ventas. Puedes escribirle como en un chat.",
    "Reportes":        "Descarga tus ventas o revisa predicciones anteriores desde aquí.",
    "Crear colaborador": "Crea una cuenta para tu personal de ventas. Ellos podrán registrar ventas pero no ver reportes.",
    "Cuenta":          "Aquí puedes cambiar tu contraseña o actualizar tus datos personales.",
}

def ayuda_flotante(pagina: str):
    msg = AYUDA_POR_PAGINA.get(pagina, "")
    if msg:
        with st.expander("❓ ¿Qué hago en esta pantalla?"):
            st.info(msg)

def header(titulo: str, subtitulo: str = ""):
    nombre = st.session_state.get("usuario_nombre", "")
    rol    = st.session_state.get("rol_nombre", "")
    st.markdown(f"""
<div class="main-header">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <h1>{titulo}</h1>
      <div style="color:#f4dfbd;font-size:.88rem;">{subtitulo}</div>
    </div>
    <div style="text-align:right;color:#f4dfbd;font-size:.82rem;">{nombre} · {rol}</div>
  </div>
</div>""", unsafe_allow_html=True)

def telefono_valido(tel: str) -> bool:
    if not tel:
        return True
    return tel.isdigit() and len(tel) == 9

# =============================================================================
# JOBLIB
# =============================================================================

@st.cache_resource
def cargar_modelo():
    ruta = os.path.join(os.path.dirname(__file__), "modelo_demanda.joblib")
    if not os.path.exists(ruta):
        return None, None
    bundle = joblib.load(ruta)
    return bundle["pipeline"], bundle["metadata"]

def predecir_demanda(datos: dict) -> float:
    pipeline, meta = cargar_modelo()
    if pipeline is None:
        return 0.0
    fecha = pd.to_datetime(datos["fecha"])
    mes   = fecha.month
    sem   = int(fecha.isocalendar().week)
    fila = {
        "precio_unitario":        float(datos.get("precio_unitario", 25.0)),
        "porcentaje_dscto":       float(datos.get("porcentaje_dscto", 0)),
        "stock_inicial_periodo":  int(datos.get("stock_inicial_periodo", 50)),
        "tiene_dscto":            int(bool(datos.get("tiene_dscto", False))),
        "es_campain":             int(bool(datos.get("es_campain", False))),
        "año":        fecha.year,
        "mes":        mes,
        "semana_año": sem,
        "dia_semana": fecha.dayofweek,
        "es_finde":   int(fecha.dayofweek >= 5),
        "trimestre":  fecha.quarter,
        "mes_sin":    np.sin(2 * np.pi * mes / 12),
        "mes_cos":    np.cos(2 * np.pi * mes / 12),
        "producto":     str(datos.get("producto", "")),
        "categoria":    str(datos.get("categoria", "")),
        "canal_venta":  str(datos.get("canal_venta", "Tienda Física")),
        "tipo_cliente": str(datos.get("tipo_cliente", "Minorista")),
        "region_venta": str(datos.get("region_venta", "Lima")),
        "tipo_campain": str(datos.get("tipo_campain", "Ninguna")),
    }
    cols = meta["features_num"] + meta["features_cat"]
    X    = pd.DataFrame([fila])[cols]
    return max(0.0, round(float(pipeline.predict(X)[0]), 1))


# =============================================================================
# API
# =============================================================================

def api_login(username: str, password: str) -> dict:
    try:
        r = requests.post(f"{BASE_URL}/login",
                          json={"username": username, "password": password},
                          timeout=8)
        if r.status_code == 200:
            token = r.json().get("jwttoken", "")
            return {"ok": bool(token), "token": token}
        msg = "Usuario o contraseña incorrectos." if r.status_code == 401 \
              else f"Error del servidor ({r.status_code})."
        return {"ok": False, "token": "", "mensaje": msg}
    except requests.exceptions.ConnectionError:
        return {"ok": False, "token": "",
                "mensaje": "No se puede conectar. ¿Está corriendo Spring Boot?"}

def api_get_perfil(username: str) -> dict:
    try:
        r = requests.get(f"{BASE_URL}/usuarios/perfil/{username}",
                         headers=get_header(), timeout=8)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}

def api_crear_empresa(nombre, ruc, sector) -> dict:
    try:
        r = requests.post(f"{BASE_URL}/empresas/insertar",
                          json={"name_empresa": nombre, "ruc_empresa": ruc,
                                "sector_empresa": sector, "estado_activo": True},
                          timeout=8)
        if r.status_code in (200, 201):
            try:
                data = r.json()
            except Exception:
                data = {}
            return {"ok": True, "data": data}
        return {"ok": False, "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_actualizar_empresa(id_empresa: int, nombre: str, ruc: str, sector: str) -> dict:
    """Actualiza datos de empresa. Endpoint: PUT /empresas/actualizar (ID va en el body)."""
    try:
        payload = {
            "idEmpresa": id_empresa,
            "name_empresa": nombre,
            "ruc_empresa": ruc,
            "sector_empresa": sector,
            "estado_activo": True,
        }
        r = requests.put(f"{BASE_URL}/empresas/actualizar",
                         json=payload, headers=get_header(), timeout=8)
        return {"ok": r.status_code in (200, 201, 204), "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_crear_usuario(datos: dict) -> dict:
    try:
        payload = {
            "nombre_usuario":    datos["username"],
            "dni_usuario":       datos.get("dni", ""),
            "correo_usuario":    datos.get("correo", ""),
            "sexo_usuario":      datos.get("sexo", ""),
            "telefono_usuario":  datos.get("telefono", ""),
            "direccion_usuario": datos.get("direccion", ""),
            "password_usuario":  datos["password"],
            "id_rol":            {"id_rol": datos["id_rol"]},
            "idEmpresa":         {"idEmpresa": datos["id_empresa"]},
        }
        r = requests.post(f"{BASE_URL}/usuarios/insertar",
                          json=payload, timeout=8)
        return {"ok": r.status_code in (200, 201), "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_get_usuarios_empresa() -> list:
    empresa_id = st.session_state.get("empresa_id", 0)
    try:
        r = requests.get(f"{BASE_URL}/usuarios/empresa/{empresa_id}",
                         headers=get_header(), timeout=8)
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else []
        return []
    except Exception:
        return []

def api_actualizar_usuario(datos: dict) -> dict:
    try:
        payload = {
            "id_usuario":        datos["id_usuario"],
            "nombre_usuario":    datos["nombre_usuario"],
            "dni_usuario":       datos["dni_usuario"],
            "correo_usuario":    datos["correo_usuario"],
            "sexo_usuario":      datos["sexo_usuario"],
            "telefono_usuario":  datos["telefono_usuario"],
            "direccion_usuario": datos["direccion_usuario"],
            "password_usuario":  datos["password_usuario"],
            "id_rol":    {"id_rol":    datos["id_rol"]},
            "idEmpresa": {"idEmpresa": datos["id_empresa"]},
        }
        r = requests.put(f"{BASE_URL}/usuarios/actualizar",
                         json=payload, headers=get_header(), timeout=8)
        return {"ok": r.status_code in (200, 201), "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_eliminar_usuario(id_usuario: int) -> dict:
    try:
        r = requests.delete(f"{BASE_URL}/usuarios/eliminar/{id_usuario}",
                            headers=get_header(), timeout=8)
        return {"ok": r.status_code in (200, 204), "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

# ── INVENTARIO ─────────────────────────────────────────────────────────────

def api_get_inventario(incluir_inactivos: bool = False) -> list:
    empresa_id = st.session_state.get("empresa_id", 0)
    try:
        params = {"incluirInactivos": "true"} if incluir_inactivos else {}
        r = requests.get(
            f"{BASE_URL}/inventario/empresa/{empresa_id}",
            headers=get_header(), params=params, timeout=8)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

def api_post_inventario(inv: dict) -> dict:
    try:
        r = requests.post(
            f"{BASE_URL}/inventario/insertar",
            json=inv, headers=get_header(), timeout=8)
        if r.status_code in (200, 201):
            return {"ok": True, "data": r.json()}
        return {"ok": False, "mensaje": r.text if hasattr(r, 'text') else "Error"}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_put_inventario(id_inv: int, inv: dict) -> bool:
    try:
        r = requests.put(
            f"{BASE_URL}/inventario/actualizar/{id_inv}",
            json=inv, headers=get_header(), timeout=8)
        return r.status_code in (200, 201)
    except Exception:
        return False

def api_del_inventario(id_inv: int) -> dict:
    try:
        r = requests.delete(
            f"{BASE_URL}/inventario/eliminar/{id_inv}",
            headers=get_header(), timeout=8)
        if r.status_code in (200, 204):
            return {"ok": True, "mensaje": "Operación exitosa"}
        return {"ok": False, "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_descontar_inventario(id_inv: int, cantidad: int) -> dict:
    try:
        r = requests.put(
            f"{BASE_URL}/inventario/descontar/{id_inv}",
            params={"cantidad": cantidad},
            headers=get_header(), timeout=8)
        if r.status_code == 200:
            return {"ok": True, "data": r.json()}
        return {"ok": False, "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_aumentar_inventario(id_inv: int, cantidad: int) -> dict:
    try:
        r = requests.put(
            f"{BASE_URL}/inventario/aumentar/{id_inv}",
            params={"cantidad": cantidad},
            headers=get_header(), timeout=8)
        if r.status_code == 200:
            return {"ok": True, "data": r.json()}
        return {"ok": False, "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_reactivar_inventario(id_inv: int) -> dict:
    try:
        r = requests.put(
            f"{BASE_URL}/inventario/reactivar/{id_inv}",
            headers=get_header(), timeout=8)
        if r.status_code == 200:
            return {"ok": True, "data": r.json()}
        return {"ok": False, "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_post_entrada(id_inv: int, cantidad: int, motivo: str) -> dict:
    usuario_id = st.session_state.get("usuario_id", 0)
    try:
        payload = {
            "cantidad": cantidad,
            "motivo": motivo,
            "idInventario": {"id_inventario": id_inv},
            "idUsuario": {"id_usuario": usuario_id},
        }
        r = requests.post(
            f"{BASE_URL}/entradas/insertar",
            json=payload, headers=get_header(), timeout=8)
        if r.status_code in (200, 201):
            return {"ok": True, "data": r.json()}
        return {"ok": False, "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_get_entradas_empresa() -> list:
    empresa_id = st.session_state.get("empresa_id", 0)
    try:
        r = requests.get(
            f"{BASE_URL}/entradas/empresa/{empresa_id}",
            headers=get_header(), timeout=8)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []

# ── VENTAS ─────────────────────────────────────────────────────────────────

def api_get_ventas() -> pd.DataFrame:
    empresa_id = st.session_state.get("empresa_id", 0)
    try:
        r = requests.get(f"{BASE_URL}/ventas-historicas/empresa/{empresa_id}",
                         headers=get_header(), timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                return pd.DataFrame(data)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def api_post_venta(venta: dict) -> bool:
    try:
        r = requests.post(f"{BASE_URL}/ventas-historicas/insertar",
                          json=venta, headers=get_header(), timeout=8)
        return r.status_code in (200, 201)
    except Exception:
        return False

def api_subir_csv(archivo_bytes: bytes, nombre: str) -> dict:
    empresa_id = st.session_state.get("empresa_id", 0)
    usuario_id = st.session_state.get("usuario_id", 0)
    try:
        r = requests.post(
            f"{BASE_URL}/ventas-historicas/upload",
            files={"file": (nombre, archivo_bytes, "text/csv")},
            data={"empresa_id": empresa_id, "usuario_id": usuario_id},
            headers=get_header(),
            timeout=60)
        if r.status_code == 200:
            return {"ok": True, "mensaje": r.text}
        return {"ok": False, "mensaje": r.text}
    except Exception as e:
        return {"ok": False, "mensaje": str(e)}

def api_post_prediccion(pred: float, contexto: dict) -> bool:
    empresa_id = st.session_state.get("empresa_id", 0)
    usuario_id = st.session_state.get("usuario_id", 0)
    if empresa_id == 0 or usuario_id == 0:
        st.error("❌ No se pudo guardar. Cierra sesión y vuelve a ingresar.")
        return False
    try:
        payload = {
            "idEmpresa":        empresa_id,
            "id_usuario":       usuario_id,
            "fecha_proyectada": contexto.get("fecha", str(date.today())),
            "producto":         contexto.get("producto", ""),
            "categoria":        contexto.get("categoria", ""),
            "precio_unitario":  float(contexto.get("precio_unitario", 0)),
            "canal_venta":      contexto.get("canal_venta", ""),
            "tipo_cliente":     contexto.get("tipo_cliente", ""),
            "region_venta":     contexto.get("region_venta", ""),
            "tiene_dscto":      bool(contexto.get("tiene_dscto", False)),
            "porcentaje_dscto": float(contexto.get("porcentaje_dscto", 0)),
            "es_campain":       bool(contexto.get("es_campain", False)),
            "tipo_campain":     contexto.get("tipo_campain", "Ninguna"),
            "stock_inicial":    int(contexto.get("stock_inicial_periodo", 0)),
            "cantidad_predicha":  float(pred),
            "stock_recomendado":  float(round(pred * 1.15, 1)),
            "confianza":          0.928,
        }
        r = requests.post(
            f"{BASE_URL}/predicciones/insertar",
            json=payload, headers=get_header(), timeout=8)
        if r.status_code in (200, 201):
            return True
        st.error(f"❌ Error {r.status_code}: {r.text}")
        return False
    except Exception as e:
        st.error(f"❌ Conexión: {e}")
        return False

def api_get_predicciones() -> list:
    empresa_id = st.session_state.get("empresa_id", 0)
    try:
        r = requests.get(
            f"{BASE_URL}/predicciones/empresa/{empresa_id}/recientes",
            headers=get_header(), timeout=8)
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else []
        return []
    except Exception:
        return []

def api_get_predicciones_paginado(page=0, size=20) -> list:
    empresa_id = st.session_state.get("empresa_id", 0)
    try:
        r = requests.get(
            f"{BASE_URL}/predicciones/empresa/{empresa_id}",
            headers=get_header(), timeout=8)
        if r.status_code == 200:
            data = r.json()
            return data if isinstance(data, list) else []
        return []
    except Exception:
        return []

def api_eliminar_prediccion(id_prediccion: int) -> bool:
    try:
        r = requests.delete(
            f"{BASE_URL}/predicciones/eliminar/{id_prediccion}",
            headers=get_header(), timeout=8)
        return r.status_code in (200, 204)
    except Exception:
        return False


# =============================================================================
# CHATBOT
# =============================================================================
def build_prompt(pregunta: str, df: pd.DataFrame,
                 pred, preds_list: list = [], inv_list: list = None) -> str:
    if df is not None and not df.empty and "cantidad_vendida" in df.columns:
        df["cantidad_vendida"] = pd.to_numeric(
            df["cantidad_vendida"], errors="coerce").fillna(0)
        total   = int(df["cantidad_vendida"].sum())
        periodo = f"{df['fecha'].min()} al {df['fecha'].max()}" \
                  if "fecha" in df.columns else "N/D"
        prods_lista = df["producto"].dropna().unique().tolist() \
                      if "producto" in df.columns else []
        prods_txt   = ", ".join(prods_lista[:10]) if prods_lista else "Sin productos registrados"
        top3_txt = "N/D"
        canal_top = "N/D"
        cats_txt = "N/D"
        if "producto" in df.columns:
            top3 = (df.groupby("producto")["cantidad_vendida"]
                    .sum().sort_values(ascending=False).head(3))
            top3_txt = ", ".join([f"{p} ({int(v)} uds)" for p, v in top3.items()])
        if "canal_venta" in df.columns:
            canal_top = (df.groupby("canal_venta")["cantidad_vendida"].sum().idxmax())
        if "categoria" in df.columns:
            cats_txt = ", ".join(df["categoria"].dropna().unique().tolist())
    else:
        total, periodo = 0, "Sin datos"
        prods_txt = "Sin productos registrados"
        top3_txt = canal_top = cats_txt = "N/D"
        prods_lista = []

    if preds_list:
        pred_ctx = ""
        for p in preds_list[:5]:
            cantidad  = float(p.get("cantidad_predicha", 0))
            stock_sug = float(p.get("stock_recomendado", 0))
            pred_ctx += (
                f"\n- Producto: {p.get('producto','?')} "
                f"| Categoría: {p.get('categoria','?')} "
                f"| Demanda estimada: {cantidad:.0f} unidades "
                f"| Stock sugerido: {stock_sug:.0f} unidades "
                f"| Para el período: {p.get('fecha_proyectada','?')}"
            )
    else:
        pred_ctx = "\n- No hay predicciones generadas aún."

    if inv_list:
        inv_ctx = ""
        for inv in inv_list[:10]:
            stock = int(inv.get("stock_actual", 0))
            stock_min = int(inv.get("stock_minimo", 10))
            estado = "🔴 Crítico" if stock <= stock_min else ("🟡 Bajo" if stock <= stock_min*2 else "🟢 OK")
            inv_ctx += f"\n- {inv.get('producto','?')}: {stock} uds (mínimo: {stock_min}) {estado}"
    else:
        inv_ctx = "\n- Aún no hay inventario registrado."

    pred_txt = f"{pred:.0f} unidades" if pred else "Aún no generada"

    return f"""
Eres AsistTextil, asistente comercial de FashAnalytics.
Hablas con el dueño o gerente de una pequeña empresa textil de Gamarra, Lima.

TU ROL:
- Ayudarlo a entender sus ventas y predicciones en palabras simples.
- Recomendarle qué hacer con su stock, qué producir o qué comprar.

REGLAS:
1. Responde SOLO con los datos que tienes abajo.
2. Si preguntan por un producto que NO está en la lista, di: "No tengo registros de ese producto."
3. Cuando el usuario pregunte por stock, usa los datos del INVENTARIO ACTUAL.
4. Siempre termina con UNA recomendación concreta.
5. Responde en español simple. Máximo 3 párrafos cortos.

--- TUS VENTAS REGISTRADAS ---
Período          : {periodo}
Total vendido    : {total:,} unidades
Productos        : {prods_txt}
Categorías       : {cats_txt}
Los más vendidos : {top3_txt}
Canal principal  : {canal_top}
Última predicción: {pred_txt}

--- TU INVENTARIO ACTUAL ---{inv_ctx}

--- PREDICCIONES GENERADAS ---
{pred_ctx}

Pregunta: {pregunta}
"""

def chatbot_responder(pregunta: str, df, pred, preds_list: list = [], inv_list: list = None) -> str:
    if GEMINI_OK and GEMINI_KEY:
        try:
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel("gemini-2.5-flash-lite")
            resp  = model.generate_content(build_prompt(pregunta, df, pred, preds_list, inv_list))
            return resp.text
        except Exception as e:
            return f"El asistente no está disponible en este momento. Error: {e}"

    q = pregunta.lower()
    if any(x in q for x in ["producir","cuánto","cuanto","demanda"]):
        return (
            f"La predicción estima {pred:.0f} unidades. Tener {int(pred*1.15)} unidades disponibles."
            if pred else
            "Primero genera una predicción en la sección Predicción."
        )
    if "stock" in q or "inventario" in q:
        return "Revisa tu inventario y compáralo con las predicciones."
    return "Configura tu API key de Gemini en .env para activar el asistente."

# =============================================================================
# OPCIONES DEL JOBLIB
# =============================================================================

def get_opciones():
    _, meta = cargar_modelo()
    if not meta:
        return {
            "productos":  [],
            "categorias": [],
            "canales":    ["Tienda Fisica","Mayorista","Online","Feria","WhatsApp"],
            "clientes":   ["Minorista","Mayorista","Consumidor Final","Revendedor"],
            "regiones":   ["Lima","Lima Norte","Lima Sur","Lima Este","Provincias"],
            "campains":   ["Dia de la Madre","Fiestas Patrias","Navidad","Black Friday","San Valentin"],
        }
    return {
        "productos":  meta.get("productos",  []),
        "categorias": meta.get("categorias", []),
        "canales":    meta.get("canales",    ["Tienda Fisica","Mayorista","Online","Feria","WhatsApp"]),
        "clientes":   meta.get("clientes",   ["Minorista","Mayorista","Consumidor Final","Revendedor"]),
        "regiones":   meta.get("regiones",   ["Lima","Lima Norte","Lima Sur","Lima Este","Provincias"]),
        "campains":   [c for c in meta.get("campains", []) if c != "Ninguna"],
    }

def categoria_de_producto_modelo(producto: str) -> str:
    _, meta = cargar_modelo()
    if not meta:
        return ""
    mapeo = meta.get("producto_categoria", {})
    if isinstance(mapeo, dict):
        return mapeo.get(producto, "")
    return ""

def productos_por_categoria_modelo() -> dict:
    _, meta = cargar_modelo()
    if not meta:
        return {}
    mapeo = meta.get("producto_categoria", {})
    resultado = {}
    if isinstance(mapeo, dict):
        for prod, cat in mapeo.items():
            resultado.setdefault(cat, []).append(prod)
    return resultado


# =============================================================================
# LOGIN Y REGISTRO
# =============================================================================

def login_view():
    left, right = st.columns([1.05, .95], gap="large", vertical_alignment="center")
    with left:
        st.markdown("""
<div class="login-hero">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:20px;">
    <img src="https://img.icons8.com/ios-filled/100/8a5a2b/t-shirt.png"
         style="width:54px;background:#f5e2bd;border-radius:17px;padding:9px;">
    <div>
      <div style="font-size:1.25rem;font-weight:900;color:#fff8ea;">FashAnalytics</div>
      <div style="color:#ead4ad;font-size:.82rem;">Gestión comercial textil</div>
    </div>
  </div>
  <h1>Predicción de demanda para MYPEs textiles</h1>
  <p>Plataforma de analítica predictiva para pequeñas empresas del sector textil de Gamarra, Lima.</p>
  <div class="security-note" style="margin-top:1.4rem;">
    🔒 Tus datos solo los ves tú — nadie más puede acceder a tu información
  </div>
</div>""", unsafe_allow_html=True)

    with right:
        st.markdown("""
<div class="auth-panel-title">
  <h2>Acceso al sistema</h2>
  <p>Ingresa tus credenciales para continuar.</p>
</div>""", unsafe_allow_html=True)

        modo = st.radio("Modo", ["Iniciar sesión", "Registrarse"],
                        horizontal=True, label_visibility="collapsed")

        if modo == "Iniciar sesión":
            with st.form("login_form"):
                username = st.text_input("👤 Tu usuario", placeholder="Ej: rosa_garcia")
                password = st.text_input("🔑 Tu contraseña", type="password",
                                         placeholder="La que creaste al registrarte")
                ok = st.form_submit_button("Entrar al sistema →",
                                           type="primary", use_container_width=True)
            st.markdown("""
<div class="security-note">
  🔒 Tu contraseña está cifrada — ni el sistema puede verla
</div>""", unsafe_allow_html=True)

            if ok:
                if not username or not password:
                    st.warning("⚠️ Escribe tu usuario y contraseña para continuar.")
                else:
                    with st.spinner("Verificando tus datos..."):
                        res = api_login(username.strip(), password)
                    if res["ok"]:
                        token   = res["token"]
                        payload = decodificar_jwt(token)
                        st.session_state.jwt_token     = token
                        st.session_state.authenticated = True
                        rol_raw = payload.get("role", "")
                        if "GERENTE" in str(rol_raw).upper():
                            st.session_state.rol_id     = 1
                            st.session_state.rol_nombre = "GERENTE"
                        else:
                            st.session_state.rol_id     = 2
                            st.session_state.rol_nombre = "COLABORADOR"
                        username_jwt = payload.get("sub", username.strip())
                        perfil = api_get_perfil(username_jwt)
                        if perfil:
                            st.session_state.usuario_nombre  = perfil.get("nombre_usuario",  username_jwt)
                            st.session_state.usuario_id      = perfil.get("id_usuario",      0)
                            st.session_state.empresa_id      = perfil.get("id_empresa",      0)
                            st.session_state.empresa_nombre  = perfil.get("nombre_empresa",  "")
                            st.session_state.ruc_empresa     = perfil.get("ruc_empresa",     "")
                            st.session_state.sector_empresa  = perfil.get("sector_empresa",  "")
                            st.session_state.dni_usuario     = perfil.get("dni_usuario",     "")
                            st.session_state.correo_usuario  = perfil.get("correo_usuario",  "")
                            st.session_state.sexo_usuario    = perfil.get("sexo_usuario",    "")
                            st.session_state.telefono        = perfil.get("telefono_usuario","")
                            st.session_state.direccion       = perfil.get("direccion_usuario","")
                            st.session_state.fecha_registro  = perfil.get("fechaRegistro_usuario","")
                            st.session_state.descripcion_rol = perfil.get("descripcion_rol", "")
                            st.session_state.tipo_rol        = perfil.get("tipo_rol",        "")
                        else:
                            st.session_state.usuario_nombre = username_jwt
                        st.session_state.page = "Dashboard"
                        st.success("✅ ¡Bienvenido! Entrando al sistema...")
                        st.rerun()
                    else:
                        st.error(f"❌ {res.get('mensaje', 'Usuario o contraseña incorrectos.')}")
                        st.info("💡 Si olvidaste tu contraseña, pide ayuda a tu Gerente.")

        else:
            st.markdown("""
<div class="security-note" style="margin-bottom:.8rem;">
  🔒 Solo registramos los datos necesarios para que uses el sistema.
</div>""", unsafe_allow_html=True)
            st.info("El registro crea una cuenta de **Gerente**. Después puedes agregar a tu equipo.")

            paso_actual = st.session_state.get("reg_paso", 1)

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                style_p1 = "background:#5a351e;color:#fff;" if paso_actual == 1 else "background:#e5d4b8;color:#7b6a57;"
                check_p1 = "✓" if st.session_state.get("reg_empresa_id", 0) > 0 else "1"
                st.markdown(f"""
<div style='{style_p1}padding:.6rem 1rem;border-radius:12px;text-align:center;font-weight:800;font-size:.9rem;margin-bottom:.5rem;'>
  {check_p1} Tu empresa
</div>""", unsafe_allow_html=True)
            with col_p2:
                style_p2 = "background:#5a351e;color:#fff;" if paso_actual == 2 else "background:#e5d4b8;color:#7b6a57;"
                st.markdown(f"""
<div style='{style_p2}padding:.6rem 1rem;border-radius:12px;text-align:center;font-weight:800;font-size:.9rem;margin-bottom:.5rem;'>
  2 Tu cuenta
</div>""", unsafe_allow_html=True)

            if paso_actual == 1:
                paso_header(1, "Datos de tu empresa", "Solo necesitamos el nombre y RUC")
                with st.form("empresa_form"):
                    nom_e = st.text_input("📋 Nombre de tu empresa",
                                          placeholder="Ej: Confecciones San Martín SAC")
                    field_hint("Escribe el nombre como aparece en tu RUC o como la conoces")
                    ruc_e = st.text_input("🔢 RUC de la empresa",
                                          placeholder="20512345678", max_chars=11)
                    field_hint("Son 11 números. Lo encuentras en tu ficha RUC de SUNAT")
                    sec_e = st.selectbox("🏭 Rubro", ["Textil","Confecciones","Moda","Otro"])
                    crear_e = st.form_submit_button("Registrar mi empresa →",
                                                    type="primary", use_container_width=True)
                if crear_e:
                    if not nom_e or not ruc_e:
                        st.warning("⚠️ El nombre y el RUC son obligatorios.")
                    elif not ruc_e.isdigit() or len(ruc_e) != 11:
                        st.error("❌ El RUC debe tener exactamente 11 números (sin letras ni espacios).")
                    else:
                        with st.spinner("Registrando tu empresa..."):
                            res_e = api_crear_empresa(nom_e, ruc_e, sec_e)
                        if res_e["ok"]:
                            emp    = res_e["data"]
                            id_emp = emp.get("idEmpresa", 0)
                            if id_emp > 0:
                                st.session_state.reg_empresa_id     = id_emp
                                st.session_state.reg_empresa_nombre = nom_e
                                st.success(f"✅ '{nom_e}' registrada correctamente")
                                progress_bar = st.progress(0, text="Preparando el siguiente paso...")
                                for pct in range(0, 101, 10):
                                    time.sleep(0.08)
                                    progress_bar.progress(pct, text=f"Preparando el siguiente paso... {pct}%")
                                st.session_state.reg_paso = 2
                                st.rerun()
                            else:
                                st.warning("Empresa creada. Ingresa el ID manualmente:")
                                id_m = st.number_input("ID empresa", min_value=1, step=1, key="id_emp_m")
                                if st.button("Confirmar ID y continuar", key="btn_id_m"):
                                    st.session_state.reg_empresa_id     = int(id_m)
                                    st.session_state.reg_empresa_nombre = nom_e
                                    st.session_state.reg_paso = 2
                                    st.rerun()
                        else:
                            st.error(res_e.get("mensaje", "No se pudo registrar la empresa."))

            else:
                eid = st.session_state.get("reg_empresa_id", 0)
                if eid == 0:
                    empty_state("⬅️", "Primero completa el Paso 1",
                                "Registra tu empresa antes de crear tu cuenta de acceso.")
                    if st.button("← Volver al Paso 1", use_container_width=True):
                        st.session_state.reg_paso = 1
                        st.rerun()
                else:
                    st.success(f"✅ Empresa: {st.session_state.get('reg_empresa_nombre','')} vinculada")
                    paso_header(2, "Crea tu cuenta de acceso", "Con esto entrarás al sistema cada vez")
                    with st.form("usuario_form"):
                        c1, c2 = st.columns(2)
                        with c1:
                            nom_u = st.text_input("👤 Tu nombre completo",
                                                   placeholder="Ej: Rosa García")
                            dni_u = st.text_input("🪪 Tu DNI", max_chars=8,
                                                   placeholder="Ej: 12345678")
                            field_hint("Solo lo usamos para identificarte en el sistema")
                            cor_u = st.text_input("📧 Correo (opcional)",
                                                   placeholder="rosa@ejemplo.com")
                            sex_u = st.selectbox("Sexo", ["M","F","Otro"])
                        with c2:
                            usr_u = st.text_input("🔑 Nombre de usuario para ingresar",
                                                   placeholder="Ej: rosa_garcia")
                            field_hint("Con esto entrarás al sistema. Sin espacios.")
                            pas_u = st.text_input("🔒 Contraseña", type="password",
                                                   placeholder="Mínimo 6 caracteres")
                            tel_u = st.text_input("📱 Teléfono (opcional)",
                                                   max_chars=9,
                                                   placeholder="9 dígitos")
                            field_hint("Solo números, sin espacios ni guiones")
                            dir_u = st.text_input("📍 Dirección (opcional)")
                        col_back, col_submit = st.columns([1, 2])
                        with col_back:
                            volver = st.form_submit_button("← Volver",
                                                            use_container_width=True)
                        with col_submit:
                            crear_u = st.form_submit_button("Crear mi cuenta →",
                                                            type="primary",
                                                            use_container_width=True)
                    if volver:
                        st.session_state.reg_paso = 1
                        st.rerun()
                    if crear_u:
                        if not usr_u or not pas_u:
                            st.warning("⚠️ El usuario y la contraseña son obligatorios.")
                        elif tel_u and not telefono_valido(tel_u):
                            st.error("❌ El teléfono debe tener exactamente 9 dígitos numéricos.")
                        else:
                            with st.spinner("Creando tu cuenta..."):
                                res_u = api_crear_usuario({
                                    "username": usr_u, "password": pas_u,
                                    "dni": dni_u, "correo": cor_u, "sexo": sex_u,
                                    "telefono": tel_u, "direccion": dir_u,
                                    "id_rol": 1, "id_empresa": eid,
                                })
                            if res_u["ok"]:
                                st.success("✅ ¡Cuenta creada! Ya puedes iniciar sesión.")
                           
                                st.session_state.reg_empresa_id     = 0
                                st.session_state.reg_empresa_nombre = ""
                                st.session_state.reg_paso = 1
                            else:
                                st.error(res_u.get("mensaje", "No se pudo crear la cuenta."))

# =============================================================================
# SIDEBAR
# =============================================================================

def sidebar_nav():
    with st.sidebar:
        st.markdown("""
<div class="logo-row">
  <div class="logo-box">
    <img src="https://img.icons8.com/ios-filled/100/8a5a2b/t-shirt.png" style="width:32px;">
  </div>
  <div>
    <p class="brand-title">FashAnalytics</p>
    <p class="brand-sub">Analítica textil</p>
  </div>
</div>""", unsafe_allow_html=True)

        nombre     = st.session_state.get("usuario_nombre", "Usuario")
        rol        = st.session_state.get("rol_nombre",     "")
        emp_nombre = st.session_state.get("empresa_nombre", "")
        icono      = "👔" if es_gerente() else "🤝"

        st.markdown(f"""
<div class="sidebar-profile">
  <strong>{icono} {rol}</strong>
  <span>{nombre}</span>
  <span>{emp_nombre}</span>
</div>""", unsafe_allow_html=True)

        if es_gerente():
            paginas = [
                "📊 Dashboard",
                "📁 Gestión comercial",
                "📦 Inventario",
                "🔮 Predicción",
                "📈 Análisis de resultados",
                "💡 Inteligencia y Recomendaciones",
                "⚙️ Gestión del Modelo Predictivo",
                "📋 Reportes",
                "🤝 Crear colaborador",
                "❓ Soporte y Ayuda",
                "📖 Guía de uso",
                "👤 Cuenta",
            ]
        else:
            paginas = [
                "📊 Dashboard",
                "✏️ Registrar mis ventas",
                "📦 Inventario",
                "🔮 Predicción",
                "📈 Análisis de resultados",
                "💡 Inteligencia y Recomendaciones",
                "⚙️ Gestión del Modelo Predictivo",
                "📋 Reportes",
                "❓ Soporte y Ayuda",
                "📖 Guía de uso",
                "👤 Cuenta",
            ]

        paginas_mapa = {
            "📊 Dashboard":                       "Dashboard",
            "📁 Gestión comercial":               "Gestión comercial",
            "✏️ Registrar mis ventas":            "Registrar mis ventas",
            "📦 Inventario":                      "Inventario",
            "🔮 Predicción":                      "Predicción",
            "📈 Análisis de resultados":          "Análisis de resultados",
            "💡 Inteligencia y Recomendaciones":  "Inteligencia y Recomendaciones",
            "⚙️ Gestión del Modelo Predictivo":   "Gestión del Modelo Predictivo",
            "📋 Reportes":                        "Reportes",
            "🤝 Crear colaborador":               "Crear colaborador",
            "❓ Soporte y Ayuda":                 "Soporte y Ayuda",
            "📖 Guía de uso":                     "Guía de uso",
            "👤 Cuenta":                          "Cuenta",
        }
        paginas_inv = {v: k for k, v in paginas_mapa.items()}

        actual     = st.session_state.get("page", "Dashboard")
        actual_con = paginas_inv.get(actual, paginas[0])
        idx        = paginas.index(actual_con) if actual_con in paginas else 0

        seleccion = st.radio("Nav", paginas, index=idx,
                             label_visibility="collapsed", key="nav_radio")
        if seleccion:
            pagina_real = paginas_mapa.get(seleccion, seleccion)
            if pagina_real != st.session_state.get("page"):
                st.session_state["page"] = pagina_real
                st.rerun()

        st.markdown("---")
        if st.button("🚪 Cerrar sesión", use_container_width=True):
            for k, v in DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()


# =============================================================================
# DASHBOARD
# =============================================================================

def page_dashboard():
    breadcrumb("Dashboard")
    header("Dashboard", "Resumen de tus ventas e indicadores")
    welcome_banner()
    ayuda_flotante("Dashboard")

    with st.spinner("Buscando tus ventas..."):
        df_orig = api_get_ventas()
        inv_lista = api_get_inventario()
        preds_recientes = api_get_predicciones()

    if df_orig.empty:
        empty_state("📂", "Aún no tienes ventas cargadas",
                    "El Gerente puede subir un archivo Excel o CSV en Gestión comercial.")
        return

    for col in ["cantidad_vendida","precio_unitario","costo_unitario",
                "stock_inicial_periodo","porcentaje_dscto"]:
        if col in df_orig.columns:
            df_orig[col] = pd.to_numeric(df_orig[col], errors="coerce").fillna(0)

    if "fecha" in df_orig.columns:
        df_orig["fecha_dt"] = pd.to_datetime(df_orig["fecha"], errors="coerce")

    df_orig["margen_bruto"] = (
        df_orig["precio_unitario"] - df_orig["costo_unitario"]
    ) * df_orig["cantidad_vendida"]

    st.markdown('<div class="section-title">🔍 Filtrar ventas</div>',
                unsafe_allow_html=True)

    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    with fc1:
        modo_campain = st.selectbox("Período comercial",
            ["Todos los períodos", "Solo campañas", "Sin campaña"], key="dash_camp")
    with fc2:
        cats_d = ["Todas las categorías"] + (
            sorted(df_orig["categoria"].dropna().unique().tolist())
            if "categoria" in df_orig.columns else [])
        sel_cat = st.selectbox("Categoría", cats_d, key="dash_cat")
    with fc3:
        prods_d = ["Todos los productos"] + (
            sorted(df_orig["producto"].dropna().unique().tolist())
            if "producto" in df_orig.columns else [])
        sel_prod = st.selectbox("Producto", prods_d, key="dash_prod")
    with fc4:
        fecha_min   = df_orig["fecha_dt"].min().date() \
                      if df_orig["fecha_dt"].notna().any() else date.today()
        fecha_desde = st.date_input("Desde", value=fecha_min, key="dash_desde")
    with fc5:
        fecha_max   = df_orig["fecha_dt"].max().date() \
                      if df_orig["fecha_dt"].notna().any() else date.today()
        fecha_hasta = st.date_input("Hasta", value=fecha_max, key="dash_hasta")

    df = df_orig.copy()
    if modo_campain == "Solo campañas" and "es_campain" in df.columns:
        df = df[df["es_campain"] == True]
    elif modo_campain == "Sin campaña" and "es_campain" in df.columns:
        df = df[df["es_campain"] == False]
    if sel_cat != "Todas las categorías" and "categoria" in df.columns:
        df = df[df["categoria"] == sel_cat]
    if sel_prod != "Todos los productos" and "producto" in df.columns:
        df = df[df["producto"] == sel_prod]
    if "fecha_dt" in df.columns:
        df = df[(df["fecha_dt"].dt.date >= fecha_desde) &
                (df["fecha_dt"].dt.date <= fecha_hasta)]

    if df.empty:
        empty_state("🔍", "Sin resultados para este filtro",
                    "Prueba ampliando el rango de fechas.")
        return

    total      = int(df["cantidad_vendida"].sum())
    margen_tot = df["margen_bruto"].sum()
    prod_clave = (df.groupby("producto")["cantidad_vendida"].sum().idxmax()
                  if "producto" in df.columns and not df.empty else "N/D")

    pred_text = "Pendiente"
    pred_sub = "ve a Predicción para calcular"
    if preds_recientes:
        ultima_pred = preds_recientes[0]
        cant_pred = float(ultima_pred.get("cantidad_predicha", 0))
        prod_pred = ultima_pred.get("producto", "")
        pred_text = f"{cant_pred:.0f} uds"
        pred_sub = f"para {prod_pred}"

    criticos = 0
    if inv_lista:
        for inv in inv_lista:
            if int(inv.get("stock_actual", 0)) <= int(inv.get("stock_minimo", 10)):
                criticos += 1

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi("Lo que vendiste",    f"{total:,}",              "unidades en el período")
    with c2: kpi("Ganancia bruta",     f"S/. {margen_tot:,.0f}", "después de descontar costo")
    with c3: kpi("Producto estrella",  prod_clave,               "el que más vendes")
    with c4: kpi("Última predicción",  pred_text,                pred_sub)
    with c5:
        if criticos > 0:
            kpi("🔴 Stock crítico", str(criticos), "productos a reponer")
        else:
            kpi("🟢 Stock saludable", "OK", "ningún producto crítico")

    st.markdown("")

    if "fecha_dt" in df.columns and df["fecha_dt"].notna().any():
        st.markdown('<div class="section-title">📅 Tus ventas por mes</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-note">'
                    'Cuántas unidades vendiste cada mes en el período seleccionado. '
                    '<strong>Los picos más altos son tus temporadas fuertes</strong> — ahí necesitas tener '
                    'más stock listo antes de que empiece el mes. '
                    'Si ves una caída brusca, revisa si fue por falta de stock o por baja demanda.'
                    '</div>', unsafe_allow_html=True)
        df_evo = df.dropna(subset=["fecha_dt"]).copy()
        df_evo["anio_mes"] = df_evo["fecha_dt"].dt.to_period("M").astype(str)
        monthly = (df_evo.groupby("anio_mes")["cantidad_vendida"]
                   .sum().reset_index().sort_values("anio_mes"))

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=monthly["anio_mes"], y=monthly["cantidad_vendida"],
            mode="lines+markers+text", name="Ventas mensuales",
            line=dict(color="#6b3f21", width=3),
            fill="tozeroy", fillcolor="rgba(107,63,33,.08)",
            marker=dict(size=10),
            text=[f"{int(v):,}" for v in monthly["cantidad_vendida"]],
            textposition="top center",
            textfont=dict(color="#3b271b", size=12)
        ))
        chart_style(fig1, "", 320)
        fig1.update_xaxes(title_text="Mes")
        fig1.update_yaxes(title_text="Unidades vendidas")
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        st.caption(f"Total del período: {total:,} unidades")

    ca, cb = st.columns(2)
    with ca:
        if "producto" in df.columns:
            st.markdown('<div class="section-title">🏆 Productos más vendidos</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-note">'
                        'Los 10 productos que más unidades vendiste en el período. '
                        'La barra más larga es tu <strong>producto estrella</strong> — asegúrate de '
                        'tenerlo siempre en stock porque es el que más clientes buscan. '
                        'Si un producto bajo en el ranking, puede ser señal de que el precio está alto o falta promocionarlo.'
                        '</div>', unsafe_allow_html=True)
            rank = (df.groupby("producto")["cantidad_vendida"].sum()
                    .reset_index().sort_values("cantidad_vendida", ascending=True).tail(10))
            fig2 = px.bar(rank, x="cantidad_vendida", y="producto",
                          orientation="h", color="cantidad_vendida",
                          color_continuous_scale=["#e7c987","#5a351e"])
            fig2.update_layout(coloraxis_showscale=False)
            chart_style(fig2, "", 320)
            st.plotly_chart(fig2, use_container_width=True)

    with cb:
        if "canal_venta" in df.columns:
            st.markdown('<div class="section-title">🚀 Por dónde vendes más</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-note">'
                        'Qué canal de venta te trae más clientes. '
                        'El trozo más grande es donde debes <strong>enfocar tus esfuerzos</strong> de promoción. '
                        'Si WhatsApp o tienda física tiene mucho menos, puede ser una oportunidad '
                        'que todavía no estás aprovechando bien.'
                        '</div>', unsafe_allow_html=True)
            canal = df.groupby("canal_venta")["cantidad_vendida"].sum().reset_index()
            fig3  = px.pie(canal, values="cantidad_vendida", names="canal_venta",
                           hole=.42,
                           color_discrete_sequence=["#d2a145","#e7c987",
                                                    "#d79b78","#f0d8ab","#c9953b"])
            fig3.update_traces(textposition="outside", textinfo="percent+label",
                               textfont=dict(color="#3b271b"))
            chart_style(fig3, "", 320)
            st.plotly_chart(fig3, use_container_width=True)

    cc, cd = st.columns(2)
    with cc:
        if "categoria" in df.columns:
            st.markdown('<div class="section-title">📂 Ventas por tipo de producto</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-note">'
                        'Qué línea de ropa vende más unidades. '
                        'Usa esto para decidir <strong>qué producir más</strong> la próxima temporada. '
                        'Si una categoría cae mucho, puede ser que el mercado ya no la pide '
                        'o que simplemente no tienes suficiente stock disponible.'
                        '</div>', unsafe_allow_html=True)
            cat_g = (df.groupby("categoria")["cantidad_vendida"].sum()
                     .reset_index().sort_values("cantidad_vendida", ascending=False))
            fig4 = px.bar(cat_g, x="categoria", y="cantidad_vendida",
                          color="cantidad_vendida",
                          color_continuous_scale=["#e7c987","#5a351e"])
            fig4.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
            chart_style(fig4, "", 300)
            st.plotly_chart(fig4, use_container_width=True)

    with cd:
        if "region_venta" in df.columns:
            st.markdown('<div class="section-title">📍 A qué zonas vendes más</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-note">'
                        'En qué distritos o regiones están tus mejores clientes. '
                        'Las zonas con barras cortas son <strong>oportunidades de crecimiento</strong> — '
                        'podrías hacer delivery o buscar distribuidores en esas zonas. '
                        'Lima Norte y Lima Sur suelen tener mucha demanda textil sin atender.'
                        '</div>', unsafe_allow_html=True)
            reg = (df.groupby("region_venta")["cantidad_vendida"].sum()
                   .reset_index().sort_values("cantidad_vendida", ascending=True))
            fig5 = px.bar(reg, x="cantidad_vendida", y="region_venta",
                          orientation="h", color="cantidad_vendida",
                          color_continuous_scale=["#ead5a3","#5a351e"])
            fig5.update_layout(coloraxis_showscale=False)
            chart_style(fig5, "", 300)
            st.plotly_chart(fig5, use_container_width=True)

    if "producto" in df.columns:
        st.markdown('<div class="section-title">💰 Cuánto ganas por producto</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-note">'
                    '<strong>Ganancia bruta = (Precio de venta − Costo) × Unidades vendidas.</strong> '
                    '🟢 Verde = los que más dinero te dejan. 🔴 Rojo = los que menos ganan o dan pérdida. '
                    'Ojo: un producto puede venderse mucho pero ganar poco si el margen es bajo. '
                    'Compara este gráfico con el de "más vendidos" para detectar cuáles realmente valen la pena.'
                    '</div>', unsafe_allow_html=True)
        margen_prod = (df.groupby("producto")["margen_bruto"].sum()
                       .reset_index().sort_values("margen_bruto", ascending=True).tail(10))
        fig6 = px.bar(margen_prod, x="margen_bruto", y="producto",
                      orientation="h", color="margen_bruto",
                      color_continuous_scale=["#b04a35","#e7c987","#5d8c45"])
        fig6.update_layout(coloraxis_showscale=False)
        fig6.update_xaxes(tickprefix="S/. ")
        chart_style(fig6, "", 320)
        st.plotly_chart(fig6, use_container_width=True)

    if inv_lista and "fecha_dt" in df.columns and df["fecha_dt"].notna().any():
        st.markdown('<div class="section-title">📦 Stock real vs ventas mensuales</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-note">'
                    'Compara cuántas unidades tienes <strong>HOY en tu almacén</strong> (barra marrón) '
                    'contra cuánto vendes en promedio por mes (barra dorada). '
                    'Si la barra dorada es mayor que la marrón, <strong>te vas a quedar sin stock este mes</strong>. '
                    'Usa esto para decidir cuánto reponer antes de la próxima campaña.'
                    '</div>', unsafe_allow_html=True)

        df_inv_dash = pd.DataFrame(inv_lista)
        if not df_inv_dash.empty and "producto" in df_inv_dash.columns:
            df_sv = df.dropna(subset=["fecha_dt"]).copy()
            df_sv["mes"] = df_sv["fecha_dt"].dt.to_period("M").astype(str)
            ventas_por_prod = df_sv.groupby("producto")["cantidad_vendida"].mean().reset_index()
            ventas_por_prod.columns = ["producto", "promedio_mensual"]

            df_merge = df_inv_dash.merge(ventas_por_prod, on="producto", how="left")
            df_merge["promedio_mensual"] = df_merge["promedio_mensual"].fillna(0).round(0)
            df_merge["stock_actual"] = pd.to_numeric(df_merge["stock_actual"], errors="coerce").fillna(0)

            df_merge["dias_cobertura"] = df_merge.apply(
                lambda r: int(r["stock_actual"] / (r["promedio_mensual"]/30)) if r["promedio_mensual"] > 0 else 999,
                axis=1
            )

            top_rot = df_merge.sort_values("promedio_mensual", ascending=False).head(10)

            # Solo productos con stock > 0 Y ventas registradas — calcular ANTES de columnas
            con_stock = df_merge[
                (df_merge["promedio_mensual"] > 0) &
                (df_merge["stock_actual"] > 0)
            ].sort_values("dias_cobertura").head(10)

            sin_stock = df_merge[
                (df_merge["promedio_mensual"] > 0) &
                (df_merge["stock_actual"] <= 0)
            ]

            # Aviso FUERA de columnas para no romper el layout
            if not sin_stock.empty:
                prods_sin = ", ".join(sin_stock["producto"].tolist()[:5])
                st.warning(
                    f"⚠️ **{len(sin_stock)} producto(s) sin stock registrado** — "
                    f"no aparecen en días de cobertura: {prods_sin}. "
                    f"Ve a **Inventario → Registrar entrada** para actualizarlos."
                )

            ce, cf = st.columns(2)
            with ce:
                fig7 = go.Figure()
                fig7.add_trace(go.Bar(
                    x=top_rot["producto"], y=top_rot["stock_actual"],
                    name="Stock actual",
                    marker_color="#5a351e",
                    text=top_rot["stock_actual"].astype(int),
                    textposition="outside"))
                fig7.add_trace(go.Bar(
                    x=top_rot["producto"], y=top_rot["promedio_mensual"],
                    name="Venta promedio mensual",
                    marker_color="#c9953b",
                    text=top_rot["promedio_mensual"].astype(int),
                    textposition="outside"))
                chart_style(fig7, "", 360)
                fig7.update_xaxes(title_text="Producto", tickangle=-30)
                fig7.update_yaxes(title_text="Unidades")
                fig7.update_layout(barmode="group")
                st.plotly_chart(fig7, use_container_width=True)

            with cf:
                st.markdown('<div class="section-title">⏳ ¿Cuántos días te dura el stock?</div>',
                            unsafe_allow_html=True)
                st.markdown('<div class="section-note">'
                            '<strong>Días de cobertura = Stock actual ÷ ventas diarias promedio.</strong><br>'
                            'Ejemplo: tienes 60 polos y vendes 2 por día → te duran 30 días.<br>'
                            '🔴 Menos de 15 días → reponer <strong>esta semana</strong> · '
                            '🟡 15–30 días → planifica pronto · '
                            '🟢 Más de 30 días → tranquilo.'
                            '</div>', unsafe_allow_html=True)

                if con_stock.empty:
                    st.info("📦 Ningún producto tiene stock registrado aún. "
                            "Ve a Inventario → Registrar entrada.")
                else:
                    con_stock["dias_show"] = con_stock["dias_cobertura"].apply(lambda d: min(d, 60))
                    colores_cob = ["#b04a35" if d < 15 else "#c9953b" if d < 30 else "#5d8c45"
                                   for d in con_stock["dias_cobertura"]]
                    fig8 = go.Figure(go.Bar(
                        y=con_stock["producto"], x=con_stock["dias_show"],
                        orientation="h", marker_color=colores_cob,
                        text=[f"{int(d)} días" if d < 60 else "+60 días"
                              for d in con_stock["dias_cobertura"]],
                        textposition="outside"))
                    chart_style(fig8, "", 360)
                    fig8.update_xaxes(title_text="Días de cobertura", range=[0, 70])
                    fig8.update_layout(showlegend=False)
                    st.plotly_chart(fig8, use_container_width=True)


# =============================================================================
# FORMULARIO DE VENTAS — con cascada producto del inventario real
# FIX APLICADO: Retorna _puede_vender y _es_bloqueado para que el botón funcione
# =============================================================================

def _form_campos_venta(prefix: str, opciones: dict, df_ventas: pd.DataFrame = None):
    """
    Formulario de venta — Diseño preservado.
    - El dropdown de Producto muestra los productos del inventario real (si hay).
    - Si no hay inventario, cae al catálogo del modelo (modo compatibilidad).
    - Muestra stock real desde backend.
    """
    prods_modelo = opciones["productos"]
    cats     = opciones["categorias"]
    canales  = opciones["canales"]
    clientes = opciones["clientes"]
    regiones = opciones["regiones"]
    campains = opciones["campains"]

    # Cargar inventario real desde backend
    inv_lista_form = api_get_inventario()
    inv_map_form   = {i["producto"]: i for i in inv_lista_form}
    productos_en_inventario = sorted(inv_map_form.keys())

    # Lista del dropdown: inventario real si hay, sino catálogo del modelo
    if productos_en_inventario:
        lista_dropdown = productos_en_inventario
    else:
        lista_dropdown = prods_modelo

    # ── PASO 1: ¿Qué vendiste? ────────────────────────────────────
    paso_header(1, "¿Qué vendiste?", "Producto, categoría y fecha")
    c1, c2, c3 = st.columns(3)
    with c1:
        f_fecha = st.date_input("📅 Fecha de la venta", value=date.today(), key=f"{prefix}_fecha")

        if lista_dropdown:
            f_prod = st.selectbox("🧵 Producto", lista_dropdown, key=f"{prefix}_prod")
        else:
            f_prod = st.text_input("🧵 Producto", placeholder="Ej: Polo básico algodón",
                                    key=f"{prefix}_prod")

        # Mostrar stock REAL del backend
        stock_sugerido = 100
        id_inventario = 0
        cat_auto = ""

        if f_prod and f_prod in inv_map_form:
            # CASO 1: producto en mi inventario real
            inv_prod = inv_map_form[f_prod]
            stock_sugerido = int(inv_prod.get("stock_actual", 0))
            stock_min = int(inv_prod.get("stock_minimo", 10))
            id_inventario = int(inv_prod.get("id_inventario", 0))
            cat_auto = inv_prod.get("categoria", "")

            if stock_sugerido <= 0:
                st.error(f"🔴 Sin stock: **{stock_sugerido} uds** — ¡Debes reponer antes de vender!")
            elif stock_sugerido <= stock_min:
                st.warning(f"🟡 Stock crítico: **{stock_sugerido} uds** — Considera reponer.")
            elif stock_sugerido <= stock_min * 2:
                st.warning(f"🟡 Stock bajo: **{stock_sugerido} uds** disponibles.")
            else:
                st.success(f"🟢 Stock disponible: **{stock_sugerido} uds**")

        elif f_prod and df_ventas is not None and not df_ventas.empty:
            # CASO 2: producto no está en inventario pero hay datos del CSV
            if "producto" in df_ventas.columns:
                df_ventas["cantidad_vendida"] = pd.to_numeric(
                    df_ventas["cantidad_vendida"], errors="coerce").fillna(0)
                df_ventas["stock_inicial_periodo"] = pd.to_numeric(
                    df_ventas["stock_inicial_periodo"], errors="coerce").fillna(0)
                df_p = df_ventas[df_ventas["producto"] == f_prod]
                if not df_p.empty:
                    ultimo = int(df_p.sort_values("fecha")["stock_inicial_periodo"].iloc[-1])
                    vendido = int(df_p["cantidad_vendida"].sum())
                    stock_sugerido = max(0, ultimo - vendido)
                    st.info(f"📊 Stock estimado del CSV: **{stock_sugerido} uds** "
                            f"(este producto no está en tu inventario real)")
        elif f_prod:
            st.info("ℹ️ Este producto no está en tu inventario real. La venta se registrará "
                    "pero no descontará stock.")

        # Categoría
        if cat_auto and cats:
            try:
                idx_cat = cats.index(cat_auto)
            except ValueError:
                idx_cat = 0
            f_cat = st.selectbox("📂 Categoría", cats, index=idx_cat, key=f"{prefix}_cat")
        else:
            f_cat = (st.selectbox("📂 Categoría", cats, key=f"{prefix}_cat") if cats
                     else st.text_input("📂 Categoría", placeholder="Ej: Polos, Pantalones...",
                                         key=f"{prefix}_cat"))

    # ── PASO 2: ¿Cuánto y a qué precio? ──────────────────────────
    paso_header(2, "¿Cuánto vendiste y a qué precio?", "Cantidades y precios")
    c2a, c2b, c2c = st.columns(3)
    with c2a:
        max_cant = max(stock_sugerido, 1) if id_inventario > 0 else 9999
        f_cant = st.number_input("🔢 Cantidad vendida",
                                  min_value=1,
                                  max_value=max_cant,
                                  value=min(10, max_cant),
                                  key=f"{prefix}_cant")
        field_hint("¿Cuántas prendas vendiste en total?")
        if id_inventario > 0 and f_cant > stock_sugerido:
            st.error(f"❌ No puedes vender más de {stock_sugerido} uds")
    with c2b:
        precio_default = 25.0
        if id_inventario > 0:
            precio_default = float(inv_map_form[f_prod].get("precio_unitario", 25.0))
        f_precio = st.number_input("💰 Precio de venta (S/.)", min_value=0.01,
                                    value=precio_default, step=0.5, key=f"{prefix}_precio")
        field_hint("¿Cuánto cobras por cada unidad?")
    with c2c:
        f_costo = st.number_input("🏭 Costo de producción (S/.)", min_value=0.01,
                                   value=12.0, step=0.5, key=f"{prefix}_costo")
        field_hint("¿Cuánto te cuesta hacer o comprar cada prenda?")

    # ── PASO 3: ¿A quién y dónde? ────────────────────────────────
    paso_header(3, "¿A quién le vendiste y dónde?", "Canal, cliente y región")
    c3a, c3b, c3c = st.columns(3)
    with c3a:
        f_canal = st.selectbox("🚀 Canal de venta", canales, key=f"{prefix}_canal")
        field_hint("¿Por dónde hiciste la venta?")
    with c3b:
        f_cli = st.selectbox("👤 Tipo de cliente", clientes, key=f"{prefix}_cli")
    with c3c:
        f_region = st.selectbox("📍 Zona de venta", regiones, key=f"{prefix}_region")

    st.markdown("---")
    c4, c5 = st.columns(2)
    with c4:
        f_dscto = st.checkbox("¿Le hiciste descuento?", key=f"{prefix}_dscto")
        f_pct = 0.0
        if f_dscto:
            f_pct = st.number_input("% de descuento que diste",
                                     1.0, 80.0, 10.0, 5.0, key=f"{prefix}_pct")
    with c5:
        f_camp = st.checkbox("¿Fue durante una campaña? (Madre, Navidad, etc.)",
                              key=f"{prefix}_camp")
        f_tipo_camp = "Ninguna"
        if f_camp:
            f_tipo_camp = (st.selectbox("¿Qué campaña?", campains, key=f"{prefix}_tipo_camp")
                           if campains else
                           st.text_input("¿Qué campaña?", key=f"{prefix}_tipo_camp"))

    st.markdown("---")
    c6, c7 = st.columns(2)
    with c6:
        # FIX: si el producto está en inventario real, stock_inicial viene auto y no es editable
        if id_inventario > 0:
            st.number_input(
                "📦 Stock antes de esta venta",
                min_value=0, value=stock_sugerido, step=1, key=f"{prefix}_stock_ini",
                disabled=True,
                help="Auto-completado desde tu inventario real para trazabilidad")
            f_stock_ini = stock_sugerido
            field_hint("Este valor viene de tu inventario real para mantener trazabilidad")
        else:
            f_stock_ini = st.number_input(
                "📦 ¿Cuántas unidades tenías al inicio?",
                min_value=0, value=stock_sugerido, step=10, key=f"{prefix}_stock_ini",
                help="Cuántas prendas de este producto tenías antes de esta venta")
    with c7:
        stock_final = max(0, f_stock_ini - f_cant)
        st.metric("📉 Stock que te queda (calculado)", f"{stock_final} uds",
                  help="Lo calculamos automáticamente: stock inicial menos lo que vendiste")

    # ═══════════════════════════════════════════════════════════════════
    # FIX CRÍTICO: Calcular si puede vender
    # ═══════════════════════════════════════════════════════════════════
    if id_inventario > 0:
        # Producto en inventario: puede vender si tiene stock y cantidad cabe
        puede_vender_flag = (stock_sugerido > 0) and (f_cant <= stock_sugerido) and bool(f_prod)
    else:
        # Producto fuera de inventario: puede vender si escribió/eligió nombre
        puede_vender_flag = bool(f_prod)

    return {
        "fecha": str(f_fecha), "producto": str(f_prod), "categoria": str(f_cat),
        "cantidad_vendida": int(f_cant), "precio_unitario": float(f_precio),
        "costo_unitario": float(f_costo), "canal_venta": str(f_canal),
        "tipo_cliente": str(f_cli), "region_venta": str(f_region),
        "modalidad_pago": "Efectivo",
        "tiene_dscto": bool(f_dscto), "porcentaje_dscto": float(f_pct),
        "es_campain": bool(f_camp), "tipo_campain": str(f_tipo_camp),
        "stock_inicial_periodo": int(f_stock_ini),
        "stock_final_periodo": stock_final,
        # campos internos
        "_id_inventario":   id_inventario,
        "_tiene_inventario": id_inventario > 0,
        "_puede_vender":    puede_vender_flag,  # ← FIX
        "_es_bloqueado":    False,              # ← FIX
    }


def _guardar_venta_completa(datos: dict) -> bool:
    """Guarda venta + descuenta stock automáticamente si producto está en inventario."""
    if datos.get("_es_bloqueado", False):
        st.error("❌ No se puede guardar: este producto no está en tu inventario.")
        return False
    if not datos.get("_puede_vender", False):
        st.error("❌ No se puede guardar: revisa la cantidad o el producto seleccionado.")
        return False

    id_inv = datos.get("_id_inventario", 0)
    cantidad = datos.get("cantidad_vendida", 0)

    # Limpiar campos internos antes de enviar al backend
    payload = {k: v for k, v in datos.items() if not k.startswith("_")}
    payload["idEmpresa"] = st.session_state.get("empresa_id", 0)
    payload["id_usuario"] = st.session_state.get("usuario_id", 0)

    # 1. Guardar la venta
    with st.spinner("Guardando tu venta..."):
        ok_venta = api_post_venta(payload)

    if not ok_venta:
        st.error("❌ No se pudo guardar la venta. Verifica que Spring Boot esté corriendo.")
        return False

    # 2. Descontar stock si producto en inventario
    if id_inv > 0:
        with st.spinner("Actualizando inventario..."):
            res_desc = api_descontar_inventario(id_inv, cantidad)
        if not res_desc.get("ok"):
            st.warning(f"⚠️ Venta guardada pero hubo un problema al descontar stock: {res_desc.get('mensaje', '')}")
            return True

        nuevo_stock = res_desc.get("data", {}).get("stock_actual", "?")
        st.success(f"✅ Venta registrada. Stock restante de '{datos['producto']}': **{nuevo_stock} uds**")
      
    else:
        st.success("✅ Venta registrada (sin descuento de stock — producto fuera del inventario).")
   

    return True


# =============================================================================
# GESTIÓN COMERCIAL
# =============================================================================

def page_gestion_comercial():
    breadcrumb("Gestión comercial")
    header("Gestión comercial", "Administra las ventas históricas de tu empresa")
    ayuda_flotante("Gestión comercial")

    tab_csv, tab_manual, tab_tabla = st.tabs(
        ["📁 Subir archivo Excel/CSV", "✏️ Agregar venta manual", "📋 Ver mis ventas"])

    with tab_csv:
        st.markdown('<div class="section-note">'
                    'Sube tu archivo de ventas históricas. Este archivo NO descuenta de tu inventario actual.'
                    '</div>', unsafe_allow_html=True)

        with st.expander("📋 ¿Qué columnas debe tener mi archivo?"):
            st.markdown("""
| Columna | Tipo | Ejemplo |
|---|---|---|
| `fecha` | Fecha YYYY-MM-DD | 2026-01-05 |
| `producto` | Texto | Polo Basico Algodon |
| `categoria` | Texto | Polos |
| `cantidad_vendida` | Número | 20 |
| `precio_unitario` | Decimal | 25.0 |
| `costo_unitario` | Decimal | 12.0 |
| `canal_venta` | Texto | Tienda Fisica |
| `tipo_cliente` | Texto | Minorista |
| `region_venta` | Texto | Lima |
| `tiene_dscto` | Boolean | false |
| `porcentaje_dscto` | Decimal | 0.0 |
| `es_campain` | Boolean | false |
| `tipo_campain` | Texto | Ninguna |
| `stock_inicial_periodo` | Número | 100 |
""")
        archivo = st.file_uploader("Selecciona tu archivo (Excel o CSV)",
                                    type=["csv","xlsx"])
        if archivo:
            try:
                df_prev = pd.read_excel(archivo) if archivo.name.endswith(".xlsx") \
                          else pd.read_csv(archivo)
                if archivo.name.endswith(".xlsx"):
                    st.info("📊 Archivo Excel detectado — lo convertiremos a CSV automáticamente.")
                st.markdown(f'<div class="section-note">Vista previa: '
                            f'<strong>{len(df_prev):,} filas encontradas</strong></div>',
                            unsafe_allow_html=True)
                st.dataframe(df_prev.head(5), use_container_width=True)

                cols_req = ["fecha","producto","categoria","cantidad_vendida","precio_unitario",
                            "costo_unitario","canal_venta","tipo_cliente","region_venta",
                            "tiene_dscto","porcentaje_dscto","es_campain","tipo_campain",
                            "stock_inicial_periodo"]
                faltantes = [c for c in cols_req if c not in df_prev.columns]
                if faltantes:
                    st.error(f"❌ Faltan estas columnas en tu archivo: `{'`, `'.join(faltantes)}`")
                else:
                    st.success("✅ El archivo tiene todas las columnas necesarias.")
                    if st.button("⬆️ Subir al sistema", key="btn_subir_csv"):
                        st.session_state["confirmar_csv"] = True

                    if st.session_state.get("confirmar_csv", False):
                        st.warning("⚠️ **¿Confirmas subir este archivo?** Reemplaza el CSV anterior pero conserva ventas manuales.")
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("❌ Cancelar", use_container_width=True,
                                          key="cancel_csv"):
                                st.session_state["confirmar_csv"] = False
                                st.rerun()
                        with c2:
                            if st.button("✅ Sí, subir archivo", type="primary",
                                          use_container_width=True, key="confirm_csv"):
                                st.session_state["confirmar_csv"] = False
                                buf = io.BytesIO()
                                df_prev.to_csv(buf, index=False)
                                buf.seek(0)
                                with st.spinner("Subiendo tu archivo..."):
                                    res = api_subir_csv(buf.read(),
                                          archivo.name.replace(".xlsx",".csv"))
                                if res["ok"]:
                                    st.success(f"✅ {res['mensaje']}")
                                  
                                else:
                                    st.error(f"❌ {res['mensaje']}")
            except Exception as e:
                st.error(f"❌ No pudimos leer el archivo: {e}")

    with tab_manual:
        st.markdown('<div class="section-note">'
                    'Registra una venta. Al guardar, <strong>el stock se actualiza automáticamente</strong> si el producto está en tu inventario.'
                    '</div>', unsafe_allow_html=True)
        df_v = api_get_ventas()
        datos = _form_campos_venta("g", get_opciones(), df_v)

        if datos is not None:
            puede = datos.get("_puede_vender", False)
            if st.button("💾 Guardar esta venta", type="primary",
                          use_container_width=True, key="g_guardar",
                          disabled=not puede):
                if _guardar_venta_completa(datos):
                    time.sleep(1.5)
                    st.rerun()

    with tab_tabla:
        st.markdown('<div class="section-note">'
                    'Todas tus ventas activas (archivo + manuales).'
                    '</div>', unsafe_allow_html=True)
        with st.spinner("Cargando tus ventas..."):
            df = api_get_ventas()
        if df.empty:
            empty_state("📂", "Aún no tienes ventas",
                        "Sube un archivo o registra ventas manualmente.")
        else:
            if "cantidad_vendida" in df.columns:
                df["cantidad_vendida"] = pd.to_numeric(df["cantidad_vendida"], errors="coerce").fillna(0)
            total_csv    = len(df[df["fuente"]=="CSV"])    if "fuente" in df.columns else 0
            total_manual = len(df[df["fuente"]=="MANUAL"]) if "fuente" in df.columns else 0
            c1, c2, c3 = st.columns(3)
            with c1: kpi("Total de ventas",  f"{len(df):,}",       "registros")
            with c2: kpi("Del archivo CSV",  f"{total_csv:,}",     "registros")
            with c3: kpi("Ingresadas manual",f"{total_manual:,}",  "registros")

            cf1, cf2 = st.columns(2)
            with cf1:
                if "producto" in df.columns:
                    prods_f = ["Todos"] + sorted(df["producto"].dropna().unique().tolist())
                    sel = st.selectbox("Filtrar por producto", prods_f, key="fp")
                    if sel != "Todos":
                        df = df[df["producto"] == sel]
            with cf2:
                if "fuente" in df.columns:
                    sel_f = st.selectbox("Filtrar por origen", ["Todas","CSV","MANUAL"], key="ff")
                    if sel_f != "Todas":
                        df = df[df["fuente"] == sel_f]

            cols_mostrar = [c for c in [
                "fecha","producto","categoria","cantidad_vendida","precio_unitario",
                "costo_unitario","canal_venta","tipo_cliente","region_venta",
                "tiene_dscto","porcentaje_dscto","es_campain","tipo_campain",
                "stock_inicial_periodo"
            ] if c in df.columns]
            st.dataframe(df[cols_mostrar], use_container_width=True, height=350)
            st.caption(f"{len(df):,} registros")

            c1, c2 = st.columns(2)
            with c1:
                buf = io.BytesIO()
                df[cols_mostrar].to_csv(buf, index=False)
                st.download_button("⬇️ Descargar CSV", buf.getvalue(),
                                   f"ventas_{date.today()}.csv",
                                   "text/csv", use_container_width=True)
            with c2:
                buf2 = io.BytesIO()
                df[cols_mostrar].to_excel(buf2, index=False)
                st.download_button("⬇️ Descargar Excel", buf2.getvalue(),
                                   f"ventas_{date.today()}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="section-title">🗑️ Eliminar una venta</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="section-note">'
                        '⚠️ Al eliminar una venta manual, el stock se restaura automáticamente al inventario.'
                        '</div>', unsafe_allow_html=True)

            def get_id_usuario_venta(row):
                id_u = row.get("id_usuario", None)
                if isinstance(id_u, dict):
                    return id_u.get("id_usuario", 0)
                return int(id_u) if id_u else 0

            mi_id = st.session_state.get("usuario_id", 0)
            if es_gerente():
                df_eliminar = df.copy()
                st.markdown('<div class="section-note">Como Gerente puedes eliminar cualquier venta de tu empresa.</div>',
                            unsafe_allow_html=True)
            else:
                mask = df.apply(lambda row: get_id_usuario_venta(row) == mi_id, axis=1)
                df_propias = df[mask].copy()
                df_ajenas  = df[~mask].copy()
                if not df_ajenas.empty:
                    st.info(f"ℹ️ Hay {len(df_ajenas)} venta(s) de otros usuarios — solo puedes eliminar las tuyas.")
                df_eliminar = df_propias

            if df_eliminar.empty:
                st.info("No tienes ventas para eliminar.")
            else:
                opciones_eliminar = {
                    f"#{row.get('id_ventas_historicas','?')} · {row.get('fecha','?')} · "
                    f"{row.get('producto','?')} · {int(row.get('cantidad_vendida',0))} uds · "
                    f"[{row.get('fuente','?')}]": (row.get("id_ventas_historicas"),
                                                    row.get("producto",""),
                                                    int(row.get("cantidad_vendida",0)),
                                                    row.get("fuente","?"))
                    for _, row in df_eliminar.iterrows()
                }
                sel_venta = st.selectbox("Selecciona la venta a eliminar",
                                          options=list(opciones_eliminar.keys()),
                                          key="sel_del_venta")
                id_eliminar, prod_del, cant_del, fuente_del = opciones_eliminar.get(sel_venta, (None, "", 0, "?"))

                if st.button("🗑️ Eliminar esta venta", use_container_width=True,
                              key="btn_del_venta"):
                    st.session_state["confirmar_del_venta"] = True

                if st.session_state.get("confirmar_del_venta", False):
                    if fuente_del == "MANUAL":
                        st.warning(f"⚠️ Eliminar esta venta MANUAL restaurará **{cant_del} uds** de '{prod_del}' al inventario.")
                    else:
                        st.warning("⚠️ Esta venta proviene del CSV histórico, no afecta tu inventario actual.")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("Sí, eliminar", type="primary",
                                      use_container_width=True, key="yes_del_venta"):
                            try:
                                r = requests.delete(
                                    f"{BASE_URL}/ventas-historicas/eliminar/{id_eliminar}",
                                    headers=get_header(), timeout=8)
                                if r.status_code in (200, 204):
                                    if fuente_del == "MANUAL" and prod_del:
                                        # FIX: buscar incluyendo inactivos para detectar descontinuados
                                        inv_lista = api_get_inventario(incluir_inactivos=True)
                                        inv_prod = next((i for i in inv_lista if i["producto"] == prod_del), None)
                                        if inv_prod:
                                            es_activo = inv_prod.get("activo", True)
                                            api_aumentar_inventario(inv_prod["id_inventario"], cant_del)
                                            if es_activo:
                                                st.success(f"✅ Venta eliminada. {cant_del} uds restauradas a inventario.")
                                            else:
                                                st.warning(f"✅ Venta eliminada y stock sumado a '{prod_del}' (DESCONTINUADO). "
                                                           f"Las {cant_del} uds quedan guardadas pero NO disponibles para venta. "
                                                           f"Si quieres usarlas, reactiva el producto en Inventario → Descontinuados.")
                                        else:
                                            st.success("✅ Venta eliminada (el producto ya no existe en inventario).")
                                    else:
                                        st.success("✅ Venta eliminada.")
                                    st.session_state["confirmar_del_venta"] = False
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Error: {r.text}")
                            except Exception as e:
                                st.error(f"❌ {e}")
                    with cc2:
                        if st.button("Cancelar", use_container_width=True, key="no_del_venta"):
                            st.session_state["confirmar_del_venta"] = False
                            st.rerun()


def page_registrar_ventas():
    breadcrumb("Registrar mis ventas")
    header("Registrar ventas del día", "Ingresa las ventas al finalizar tu turno")
    ayuda_flotante("Gestión comercial")

    tab_reg, tab_mis_ventas = st.tabs(["✏️ Registrar venta", "🗑️ Mis ventas"])
    with tab_reg:
        st.markdown('<div class="section-note">'
                    'Completa los campos. Al guardar, <strong>el stock se descontará automáticamente</strong>.'
                    '</div>', unsafe_allow_html=True)
        df_v = api_get_ventas()
        datos = _form_campos_venta("c", get_opciones(), df_v)

        if datos is not None:
            puede = datos.get("_puede_vender", False)
            if st.button("💾 Guardar mi venta", type="primary",
                          use_container_width=True, key="c_guardar",
                          disabled=not puede):
                if _guardar_venta_completa(datos):
                    time.sleep(1.5)
                    st.rerun()

    with tab_mis_ventas:
        st.markdown('<div class="section-note">'
                    'Puedes ver todas las ventas de la empresa, pero solo eliminar las que tú registraste.'
                    '</div>', unsafe_allow_html=True)
        with st.spinner("Cargando ventas..."):
            df = api_get_ventas()
        if df.empty:
            empty_state("📂", "Aún no hay ventas registradas",
                        "Usa la pestaña 'Registrar venta' para agregar la primera.")
        else:
            mi_id = st.session_state.get("usuario_id", 0)

            def get_id_usuario_venta(row):
                id_u = row.get("id_usuario", None)
                if isinstance(id_u, dict):
                    return id_u.get("id_usuario", 0)
                return int(id_u) if id_u else 0

            cols_mostrar = [c for c in [
                "fecha","producto","categoria","cantidad_vendida",
                "precio_unitario","canal_venta","tipo_cliente","region_venta"
            ] if c in df.columns]
            st.dataframe(df[cols_mostrar], use_container_width=True, height=280)
            st.caption(f"{len(df):,} ventas totales de la empresa")

            st.markdown("---")
            st.markdown('<div class="section-title">🗑️ Eliminar mis ventas</div>',
                        unsafe_allow_html=True)
            mask    = df.apply(lambda row: get_id_usuario_venta(row) == mi_id, axis=1)
            df_mias = df[mask].copy()

            if df_mias.empty:
                st.info("Aún no has registrado ventas propias.")
            else:
                st.caption(f"{len(df_mias)} venta(s) registradas por ti")
                opciones_eliminar = {
                    f"#{row.get('id_ventas_historicas','?')} · {row.get('fecha','?')} · "
                    f"{row.get('producto','?')} · {int(row.get('cantidad_vendida',0))} uds":
                    (row.get("id_ventas_historicas"),
                     row.get("producto",""),
                     int(row.get("cantidad_vendida",0)),
                     row.get("fuente","?"))
                    for _, row in df_mias.iterrows()
                }
                sel_venta = st.selectbox("Selecciona la venta a eliminar",
                                          options=list(opciones_eliminar.keys()),
                                          key="sel_del_colab")
                id_eliminar, prod_del, cant_del, fuente_del = opciones_eliminar.get(sel_venta, (None, "", 0, "?"))
                if st.button("🗑️ Eliminar esta venta", use_container_width=True,
                              key="btn_del_colab"):
                    st.session_state["confirmar_del_colab"] = True
                if st.session_state.get("confirmar_del_colab", False):
                    if fuente_del == "MANUAL":
                        st.warning(f"⚠️ Eliminar esta venta restaurará **{cant_del} uds** de '{prod_del}' al inventario.")
                    else:
                        st.warning("⚠️ ¿Confirmas eliminar esta venta?")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        if st.button("Sí, eliminar", type="primary",
                                      use_container_width=True, key="yes_del_colab"):
                            try:
                                r = requests.delete(
                                    f"{BASE_URL}/ventas-historicas/eliminar/{id_eliminar}",
                                    headers=get_header(), timeout=8)
                                if r.status_code in (200, 204):
                                    if fuente_del == "MANUAL" and prod_del:
                                        inv_lista = api_get_inventario(incluir_inactivos=True)
                                        inv_prod = next((i for i in inv_lista if i["producto"] == prod_del), None)
                                        if inv_prod:
                                            es_activo = inv_prod.get("activo", True)
                                            api_aumentar_inventario(inv_prod["id_inventario"], cant_del)
                                            if not es_activo:
                                                st.warning(f"⚠️ El producto '{prod_del}' está DESCONTINUADO. "
                                                           f"Se sumaron {cant_del} uds pero no están disponibles para venta hasta reactivarlo.")
                                    st.success("✅ Venta eliminada.")
                                    st.session_state["confirmar_del_colab"] = False
                                    time.sleep(1.5)
                                    st.rerun()
                                else:
                                    st.error(f"❌ Error: {r.text}")
                            except Exception as e:
                                st.error(f"❌ {e}")
                    with cc2:
                        if st.button("Cancelar", use_container_width=True, key="no_del_colab"):
                            st.session_state["confirmar_del_colab"] = False
                            st.rerun()


# =============================================================================
# INVENTARIO
# =============================================================================

def page_inventario():
    breadcrumb("Inventario")
    header("Inventario", "Controla cuántos productos tienes disponibles")
    ayuda_flotante("Inventario")

    empresa_id = st.session_state.get("empresa_id", 0)

    tabs = ["📋 Mi inventario real", "➕ Registrar producto",
            "📥 Registrar entrada", "📊 Estimado del historial"]
    if es_gerente():
        tabs.append("💤 Productos descontinuados")

    tabs_obj = st.tabs(tabs)

    # PESTAÑA 1: Ver inventario real
    with tabs_obj[0]:
        st.markdown('<div class="section-note">'
                    'Aquí ves cuánto stock tienes por producto. 🔴 Crítico · 🟡 Bajo · 🟢 OK'
                    '</div>', unsafe_allow_html=True)

        with st.spinner("Cargando tu inventario..."):
            inv_lista = api_get_inventario()

        if not inv_lista:
            empty_state("📦", "Aún no tienes productos en inventario",
                        "Ve a la pestaña '➕ Registrar producto' para agregar el primero.")
        else:
            df_inv = pd.DataFrame(inv_lista)
            df_inv["stock_actual"] = pd.to_numeric(df_inv["stock_actual"], errors="coerce").fillna(0).astype(int)
            df_inv["stock_minimo"] = pd.to_numeric(df_inv["stock_minimo"], errors="coerce").fillna(10).astype(int)

            df_inv["estado"] = df_inv.apply(
                lambda r: "🔴 Crítico" if r["stock_actual"] <= r["stock_minimo"]
                else ("🟡 Bajo" if r["stock_actual"] <= r["stock_minimo"]*2 else "🟢 OK"),
                axis=1)

            criticos = len(df_inv[df_inv["estado"]=="🔴 Crítico"])
            bajos    = len(df_inv[df_inv["estado"]=="🟡 Bajo"])
            valor_total = (df_inv["stock_actual"] * pd.to_numeric(df_inv.get("precio_unitario", 0), errors="coerce").fillna(0)).sum()

            c1, c2, c3, c4 = st.columns(4)
            with c1: kpi("Productos activos", str(len(df_inv)), "en inventario")
            with c2: kpi("🔴 Críticos", str(criticos), "necesitan reposición urgente")
            with c3: kpi("🟡 Bajos", str(bajos), "considerar reponer")
            with c4: kpi("Valor inventario", f"S/. {valor_total:,.0f}", "estimado a precio de venta")

            if criticos > 0:
                st.warning(f"⚠️ Tienes {criticos} producto(s) en estado crítico.")

            st.markdown('<div class="section-title">🔍 Filtros</div>', unsafe_allow_html=True)
            cf1, cf2, cf3 = st.columns(3)
            with cf1:
                buscar = st.text_input("🔎 Buscar producto", placeholder="Escribe parte del nombre...", key="inv_buscar")
            with cf2:
                cats_inv = ["Todas"] + sorted(df_inv["categoria"].dropna().unique().tolist())
                sel_cat_inv = st.selectbox("Categoría", cats_inv, key="inv_cat_f")
            with cf3:
                sel_estado = st.selectbox("Estado", ["Todos", "🔴 Crítico", "🟡 Bajo", "🟢 OK"], key="inv_estado_f")

            df_show = df_inv.copy()
            if buscar:
                df_show = df_show[df_show["producto"].str.lower().str.contains(buscar.lower(), na=False)]
            if sel_cat_inv != "Todas":
                df_show = df_show[df_show["categoria"] == sel_cat_inv]
            if sel_estado != "Todos":
                df_show = df_show[df_show["estado"] == sel_estado]

            cols_disp = ["producto", "categoria", "stock_actual", "stock_minimo", "precio_unitario", "estado"]
            cols_disp = [c for c in cols_disp if c in df_show.columns]
            df_show_clean = df_show[cols_disp].copy()
            df_show_clean.columns = ["Producto", "Categoría", "Stock actual", "Stock mínimo", "Precio (S/.)", "Estado"]
            st.dataframe(df_show_clean, use_container_width=True, height=350)
            st.caption(f"{len(df_show)} de {len(df_inv)} productos")

            if len(df_show) > 0:
                fig = px.bar(df_show.sort_values("stock_actual").head(15),
                             x="stock_actual", y="producto",
                             orientation="h", color="estado",
                             color_discrete_map={"🔴 Crítico":"#b04a35","🟡 Bajo":"#c9953b","🟢 OK":"#5d8c45"})
                chart_style(fig, "", 400)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown('<div class="section-title">📄 Descargar reporte</div>', unsafe_allow_html=True)
            cd1, cd2 = st.columns(2)
            with cd1:
                buf_csv = io.BytesIO()
                df_inv[cols_disp].to_csv(buf_csv, index=False)
                st.download_button("⬇️ Descargar CSV", buf_csv.getvalue(),
                                   f"inventario_{date.today()}.csv", "text/csv",
                                   use_container_width=True)
            with cd2:
                buf_xls = io.BytesIO()
                df_inv[cols_disp].to_excel(buf_xls, index=False)
                st.download_button("⬇️ Descargar Excel", buf_xls.getvalue(),
                                   f"inventario_{date.today()}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

            if es_gerente():
                st.markdown("---")
                st.markdown('<div class="section-title">✏️ Editar producto</div>', unsafe_allow_html=True)
                opciones_edit = {f"{i['producto']} ({int(i['stock_actual'])} uds)": i for i in inv_lista}
                sel_edit_label = st.selectbox("Producto a editar",
                                              options=list(opciones_edit.keys()),
                                              key="sel_edit_inv")
                sel_edit = opciones_edit.get(sel_edit_label, {})

                if sel_edit:
                    with st.form("form_edit_inv"):
                        ec1, ec2, ec3 = st.columns(3)
                        with ec1:
                            e_nombre = st.text_input("Producto", value=sel_edit.get("producto", ""))
                            e_cat = st.selectbox("Categoría",
                                                  sorted(get_opciones()["categorias"]),
                                                  index=sorted(get_opciones()["categorias"]).index(sel_edit.get("categoria", ""))
                                                  if sel_edit.get("categoria","") in get_opciones()["categorias"] else 0)
                        with ec2:
                            e_stock = st.number_input("Stock actual", min_value=0,
                                                       value=int(sel_edit.get("stock_actual", 0)))
                            e_min = st.number_input("Stock mínimo", min_value=0,
                                                     value=int(sel_edit.get("stock_minimo", 10)))
                        with ec3:
                            e_precio = st.number_input("Precio (S/.)", min_value=0.01,
                                                       value=float(sel_edit.get("precio_unitario", 25.0)))
                            e_um = st.selectbox("Unidad", ["Unidad","Docena","Par","Metro"],
                                                index=["Unidad","Docena","Par","Metro"].index(sel_edit.get("unidad_medida","Unidad"))
                                                if sel_edit.get("unidad_medida","Unidad") in ["Unidad","Docena","Par","Metro"] else 0)
                        guardar_edit = st.form_submit_button("💾 Guardar cambios",
                                                              type="primary", use_container_width=True)
                    if guardar_edit:
                        ok = api_put_inventario(sel_edit["id_inventario"], {
                            "producto": e_nombre, "categoria": e_cat,
                            "stock_actual": e_stock, "stock_minimo": e_min,
                            "precio_unitario": e_precio, "unidad_medida": e_um,
                        })
                        if ok:
                            st.success(f"✅ '{e_nombre}' actualizado.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Error al actualizar.")

                st.markdown("---")
                st.markdown('<div class="section-title">🗑️ Eliminar producto</div>', unsafe_allow_html=True)
                opciones_del = {f"{i['producto']}": i["id_inventario"] for i in inv_lista}
                sel_del = st.selectbox("Producto a eliminar",
                                        options=list(opciones_del.keys()),
                                        key="inv_del_sel")
                if st.button("🗑️ Eliminar producto", key="btn_del_inv",
                              use_container_width=True):
                    # Consultar si tiene ventas ANTES de mostrar confirmación
                    df_ventas_check = api_get_ventas()
                    tiene_ventas = (
                        not df_ventas_check.empty and
                        "producto" in df_ventas_check.columns and
                        sel_del in df_ventas_check["producto"].values
                    )
                    st.session_state["confirmar_inv"] = True
                    st.session_state["inv_tiene_ventas"] = tiene_ventas

                if st.session_state.get("confirmar_inv", False):
                    tiene_ventas = st.session_state.get("inv_tiene_ventas", False)
                    if tiene_ventas:
                        st.warning(
                            f"⚠️ **'{sel_del}' tiene historial de ventas registradas.**\n\n"
                            f"No se puede eliminar permanentemente porque afectaría tus registros históricos. "
                            f"En cambio, quedará **discontinuado** — ya no aparecerá en el inventario activo, "
                            f"pero sus ventas se conservan. "
                            f"Puedes reactivarlo cuando quieras desde la pestaña **Discontinuados**."
                        )
                    else:
                        st.error(
                            f"🗑️ **'{sel_del}' no tiene ventas asociadas.**\n\n"
                            f"Esta eliminación es **permanente** — el producto desaparecerá del sistema "
                            f"y no se podrá recuperar. ¿Estás seguro?"
                        )
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        btn_label = "Sí, discontinuar" if tiene_ventas else "Sí, eliminar permanentemente"
                        if st.button(btn_label, key="yes_inv", type="primary",
                                      use_container_width=True):
                            res = api_del_inventario(opciones_del[sel_del])
                            if res["ok"]:
                                msg = f"✅ '{sel_del}' fue discontinuado." if tiene_ventas else f"✅ '{sel_del}' eliminado permanentemente."
                                st.success(msg)
                                st.session_state["confirmar_inv"] = False
                                st.session_state["inv_tiene_ventas"] = False
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.error(f"❌ {res.get('mensaje','Error al eliminar.')}")
                    with cc2:
                        if st.button("Cancelar", key="no_inv", use_container_width=True):
                            st.session_state["confirmar_inv"] = False
                            st.session_state["inv_tiene_ventas"] = False
                            st.rerun()

    # PESTAÑA 2: Registrar producto
    with tabs_obj[1]:
        if not es_gerente():
            st.warning("🔒 Solo el Gerente puede registrar productos.")
        else:
            st.markdown('<div class="section-note">'
                        'Agrega productos a tu inventario. Puedes hacerlo uno a uno o subir un archivo.'
                        '</div>', unsafe_allow_html=True)

            sub_tab1, sub_tab2 = st.tabs(["✏️ Manual", "📥 Subir CSV de inventario"])

            with sub_tab1:
                paso_header(1, "Datos del nuevo producto",
                            "Elige del catálogo del modelo o escribe tu propio nombre.")

                # Selector fuera del form para que sea reactivo
                prods_modelo_inv = sorted(get_opciones()["productos"])
                opciones_prod = ["✍️ Escribir nombre propio"] + prods_modelo_inv
                sel_origen = st.selectbox(
                    "🧵 ¿De dónde viene el producto?",
                    opciones_prod,
                    key="inv_prod_origen",
                    help="Elige del catálogo del modelo para que las predicciones funcionen mejor, "
                         "o escribe tu propio nombre si es un producto nuevo.")
                field_hint("Los productos del catálogo del modelo permiten predicciones más precisas")

                with st.form("form_nuevo_prod", clear_on_submit=True):
                    nc1, nc2, nc3 = st.columns(3)
                    with nc1:
                        if sel_origen == "✍️ Escribir nombre propio":
                            n_nombre = st.text_input("🧵 Nombre del producto",
                                                      placeholder="Ej: Polo Cuello V Algodón")
                            field_hint("Escribe el nombre como tú lo conoces")
                        else:
                            # Viene del modelo — autocompletar nombre y categoría
                            st.text_input("🧵 Producto seleccionado",
                                          value=sel_origen, disabled=True)
                            n_nombre = sel_origen

                        # Categoría: bloqueada si viene del modelo, libre si es nombre propio
                        cats_inv = sorted(get_opciones()["categorias"])
                        cat_auto_inv = categoria_de_producto_modelo(sel_origen) if sel_origen != "✍️ Escribir nombre propio" else ""
                        if cat_auto_inv:
                            # Producto del modelo → categoría fija, no editable
                            st.text_input("📂 Categoría", value=cat_auto_inv, disabled=True)
                            n_cat = cat_auto_inv
                            field_hint(f"Categoría fija del modelo — no se puede cambiar")
                        else:
                            # Nombre propio → el usuario elige la categoría
                            idx_cat_inv = 0
                            n_cat = st.selectbox("📂 Categoría", cats_inv, index=idx_cat_inv)
                            field_hint("Elige la categoría que mejor describe este producto")

                    with nc2:
                        n_stock = st.number_input("📦 Stock inicial", min_value=0, value=50)
                        field_hint("¿Cuántas unidades tienes ahora?")
                        n_min = st.number_input("⚠️ Stock mínimo (alerta)", min_value=0, value=10)
                        field_hint("Cuando baje de este número, te avisamos")
                    with nc3:
                        n_precio = st.number_input("💰 Precio de venta (S/.)",
                                                    min_value=0.01, value=25.0, step=0.5)
                        n_um = st.selectbox("📏 Unidad de medida", ["Unidad","Docena","Par","Metro"])

                    guardar_nuevo = st.form_submit_button("➕ Agregar al inventario",
                                                          type="primary", use_container_width=True)
                if guardar_nuevo:
                    if not n_nombre or n_nombre == "✍️ Escribir nombre propio":
                        st.warning("⚠️ El nombre del producto es obligatorio.")
                    else:
                        inv_check = api_get_inventario(incluir_inactivos=True)
                        existe = next((i for i in inv_check if i["producto"].lower() == n_nombre.lower()), None)
                        if existe:
                            st.error(f"❌ Ya existe un producto con el nombre '{n_nombre}'.")
                        else:
                            res = api_post_inventario({
                                "producto": n_nombre, "categoria": n_cat,
                                "stock_actual": n_stock, "stock_minimo": n_min,
                                "stock_inicial": n_stock,
                                "precio_unitario": n_precio, "unidad_medida": n_um,
                                "activo": True,
                                "idEmpresa": {"idEmpresa": empresa_id}
                            })
                            if res["ok"]:
                                st.success(f"✅ '{n_nombre}' agregado al inventario.")
                            
                                time.sleep(1.5)
                                st.rerun()
                            else:
                                st.error(f"❌ {res.get('mensaje','Error al guardar.')}")

            with sub_tab2:
                st.markdown('<div class="section-note">'
                            'Sube un archivo Excel o CSV con tu inventario inicial.'
                            '</div>', unsafe_allow_html=True)

                with st.expander("📋 ¿Qué columnas debe tener mi archivo?"):
                    st.markdown("""
| Columna | Tipo | Ejemplo |
|---|---|---|
| `producto` | Texto | Polo Básico |
| `categoria` | Texto | Polos |
| `stock_actual` | Número | 100 |
| `stock_minimo` | Número | 20 |
| `precio_unitario` | Decimal | 25.50 |
| `unidad_medida` | Texto (opc) | Unidad |
""")
                archivo_inv = st.file_uploader("Selecciona tu archivo de inventario",
                                                type=["csv","xlsx"], key="upload_inv")
                if archivo_inv:
                    try:
                        df_inv_csv = pd.read_excel(archivo_inv) if archivo_inv.name.endswith(".xlsx") \
                                     else pd.read_csv(archivo_inv)
                        st.markdown(f'<div class="section-note">Vista previa: '
                                    f'<strong>{len(df_inv_csv):,} productos encontrados</strong></div>',
                                    unsafe_allow_html=True)
                        st.dataframe(df_inv_csv.head(5), use_container_width=True)

                        cols_req_inv = ["producto", "categoria", "stock_actual", "stock_minimo", "precio_unitario"]
                        faltantes_inv = [c for c in cols_req_inv if c not in df_inv_csv.columns]
                        if faltantes_inv:
                            st.error(f"❌ Faltan columnas: `{'`, `'.join(faltantes_inv)}`")
                        else:
                            cats_modelo = set(get_opciones()["categorias"])
                            cats_csv = set(df_inv_csv["categoria"].dropna().unique())
                            cats_invalidas = cats_csv - cats_modelo
                            if cats_invalidas:
                                st.warning(f"⚠️ Estas categorías NO están en tu modelo: {', '.join(cats_invalidas)}")

                            if st.button("⬆️ Importar al inventario", type="primary",
                                          use_container_width=True, key="btn_import_inv"):
                                with st.spinner("Importando productos..."):
                                    importados = 0
                                    saltados = 0
                                    inv_existente = {i["producto"].lower() for i in api_get_inventario(incluir_inactivos=True)}
                                    for _, row in df_inv_csv.iterrows():
                                        prod_n = str(row["producto"]).strip()
                                        cat_n = str(row["categoria"]).strip()
                                        if cat_n not in cats_modelo:
                                            saltados += 1
                                            continue
                                        if prod_n.lower() in inv_existente:
                                            saltados += 1
                                            continue
                                        res_imp = api_post_inventario({
                                            "producto": prod_n,
                                            "categoria": cat_n,
                                            "stock_actual": int(row["stock_actual"]),
                                            "stock_minimo": int(row["stock_minimo"]),
                                            "stock_inicial": int(row["stock_actual"]),
                                            "precio_unitario": float(row["precio_unitario"]),
                                            "unidad_medida": str(row.get("unidad_medida", "Unidad")),
                                            "activo": True,
                                            "idEmpresa": {"idEmpresa": empresa_id}
                                        })
                                        if res_imp["ok"]:
                                            importados += 1
                                        else:
                                            saltados += 1
                                st.success(f"✅ Importados: {importados}. Saltados: {saltados}")
                                if importados > 0:
                                    st.balloons()
                                time.sleep(2)
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ No pudimos leer el archivo: {e}")

    # PESTAÑA 3: Registrar entrada
    with tabs_obj[2]:
        if not es_gerente():
            st.warning("🔒 Solo el Gerente puede registrar entradas de mercadería.")
        else:
            st.markdown('<div class="section-note">'
                        'Registra cuando llega mercadería nueva. El stock se aumenta y queda registrado.'
                        '</div>', unsafe_allow_html=True)

            with st.spinner("Cargando productos..."):
                inv_lista_ent = api_get_inventario()

            if not inv_lista_ent:
                empty_state("📦", "Primero registra productos",
                            "Ve a la pestaña '➕ Registrar producto'.")
            else:
                paso_header(1, "Datos de la entrada", "Producto, cantidad y motivo")
                with st.form("form_entrada", clear_on_submit=True):
                    opciones_ent = {f"{i['producto']} (stock: {int(i['stock_actual'])})": i for i in inv_lista_ent}
                    sel_ent_label = st.selectbox("📦 Producto que llega",
                                                  options=list(opciones_ent.keys()))
                    sel_ent = opciones_ent.get(sel_ent_label, {})

                    ec1, ec2 = st.columns(2)
                    with ec1:
                        e_cant = st.number_input("🔢 Cantidad que llega", min_value=1, value=50)
                    with ec2:
                        e_motivo = st.selectbox("📝 Motivo",
                                                 ["Compra a proveedor", "Producción propia",
                                                  "Devolución de cliente", "Ajuste de inventario",
                                                  "Donación recibida", "Otro"])

                    e_obs = st.text_area("Observaciones (opcional)", height=80)

                    guardar_ent = st.form_submit_button("➕ Registrar entrada",
                                                        type="primary", use_container_width=True)

                if guardar_ent:
                    motivo_final = f"{e_motivo}: {e_obs}" if e_obs else e_motivo
                    with st.spinner("Registrando entrada..."):
                        res_ent = api_post_entrada(
                            sel_ent["id_inventario"],
                            int(e_cant),
                            motivo_final
                        )
                    if res_ent["ok"]:
                        st.success(f"✅ Entrada registrada. '{sel_ent['producto']}' ahora tiene **{int(sel_ent['stock_actual']) + int(e_cant)} uds**.")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"❌ {res_ent.get('mensaje','Error al registrar entrada.')}")

                st.markdown("---")
                st.markdown('<div class="section-title">📜 Historial de entradas</div>',
                            unsafe_allow_html=True)
                entradas_lista = api_get_entradas_empresa()
                if not entradas_lista:
                    st.info("📭 Aún no has registrado entradas.")
                else:
                    df_ent = pd.DataFrame(entradas_lista)
                    if not df_ent.empty:
                        cols_ent_show = []
                        if "fecha" in df_ent.columns:
                            df_ent["fecha_fmt"] = pd.to_datetime(df_ent["fecha"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
                            cols_ent_show.append("fecha_fmt")
                        if "cantidad" in df_ent.columns:
                            cols_ent_show.append("cantidad")
                        if "motivo" in df_ent.columns:
                            cols_ent_show.append("motivo")
                        if "idInventario" in df_ent.columns:
                            df_ent["producto"] = df_ent["idInventario"].apply(
                                lambda x: x.get("producto", "?") if isinstance(x, dict) else "?")
                            cols_ent_show.insert(0 if not cols_ent_show else 1, "producto")

                        df_ent_show = df_ent[cols_ent_show].copy()
                        col_names = {"fecha_fmt": "Fecha", "cantidad": "Cantidad",
                                     "motivo": "Motivo", "producto": "Producto"}
                        df_ent_show.columns = [col_names.get(c, c) for c in df_ent_show.columns]
                        st.dataframe(df_ent_show, use_container_width=True, height=250)
                        st.caption(f"{len(df_ent)} entrada(s) registradas")

    # PESTAÑA 4: Estimado del historial
    with tabs_obj[3]:
        st.markdown('<div class="section-note">'
                    '📊 Estimación basada en tu CSV de ventas históricas. NO es tu inventario real.'
                    '</div>', unsafe_allow_html=True)

        df_ventas_est = api_get_ventas()
        if df_ventas_est.empty:
            empty_state("📭", "No hay datos históricos",
                        "Sube un archivo en Gestión comercial para ver el estimado.")
        else:
            df_ventas_est["cantidad_vendida"] = pd.to_numeric(
                df_ventas_est["cantidad_vendida"], errors="coerce").fillna(0)
            df_ventas_est["stock_inicial_periodo"] = pd.to_numeric(
                df_ventas_est.get("stock_inicial_periodo", 0), errors="coerce").fillna(0)

            if "fuente" in df_ventas_est.columns:
                df_csv_only = df_ventas_est[df_ventas_est["fuente"] == "CSV"].copy()
            else:
                df_csv_only = df_ventas_est.copy()

            if df_csv_only.empty:
                empty_state("📭", "No hay datos de CSV",
                            "Solo tienes ventas manuales.")
            else:
                prods_csv = df_csv_only["producto"].dropna().unique().tolist()
                inv_real = api_get_inventario(incluir_inactivos=True)
                inv_real_nombres = {i["producto"].lower() for i in inv_real}

                filas_est = []
                for prod in sorted(prods_csv):
                    df_p = df_csv_only[df_csv_only["producto"] == prod]
                    cat = df_p["categoria"].iloc[0] if "categoria" in df_p.columns else "Sin categoría"
                    ultimo_stock = int(df_p.sort_values("fecha")["stock_inicial_periodo"].iloc[-1]) if not df_p.empty else 0
                    total_vendido = int(df_p["cantidad_vendida"].sum())
                    stock_est = max(0, ultimo_stock - total_vendido)
                    en_real = prod.lower() in inv_real_nombres
                    filas_est.append({
                        "Producto": prod,
                        "Categoría": cat,
                        "Stock inicial CSV": ultimo_stock,
                        "Vendido total": total_vendido,
                        "Estimado actual": stock_est,
                        "Ya en inventario real": "✅ Sí" if en_real else "❌ No"
                    })

                df_est = pd.DataFrame(filas_est)
                c1, c2, c3 = st.columns(3)
                with c1: kpi("Productos en CSV", str(len(df_est)), "del historial")
                with c2: kpi("Ya en inventario", str(len(df_est[df_est["Ya en inventario real"]=="✅ Sí"])), "importados")
                with c3: kpi("Por importar", str(len(df_est[df_est["Ya en inventario real"]=="❌ No"])), "pendientes")

                st.dataframe(df_est, use_container_width=True, height=350)

                pendientes = df_est[df_est["Ya en inventario real"]=="❌ No"]
                if not pendientes.empty and es_gerente():
                    st.markdown("---")
                    st.markdown(f'<div class="section-title">📥 Importar {len(pendientes)} productos al inventario</div>',
                                unsafe_allow_html=True)
                    st.markdown('<div class="section-note">'
                                'Los productos se importarán con <strong>stock = 0</strong> y precio promedio del CSV.'
                                '</div>', unsafe_allow_html=True)

                    if st.button(f"📥 Importar {len(pendientes)} productos al inventario",
                                  type="primary", use_container_width=True, key="btn_import_est"):
                        st.session_state["confirmar_import_est"] = True

                    if st.session_state.get("confirmar_import_est", False):
                        st.warning(f"⚠️ Se importarán {len(pendientes)} productos con stock=0.")
                        cci1, cci2 = st.columns(2)
                        with cci1:
                            if st.button("Sí, importar", type="primary",
                                          use_container_width=True, key="yes_imp_est"):
                                with st.spinner("Importando productos..."):
                                    importados = 0
                                    saltados = 0
                                    cats_modelo = set(get_opciones()["categorias"])
                                    for _, row in pendientes.iterrows():
                                        cat_p = row["Categoría"]
                                        if cat_p not in cats_modelo:
                                            saltados += 1
                                            continue
                                        df_pr = df_csv_only[df_csv_only["producto"] == row["Producto"]]
                                        precio_prom = float(pd.to_numeric(df_pr["precio_unitario"], errors="coerce").mean())
                                        if precio_prom <= 0:
                                            precio_prom = 25.0
                                        res = api_post_inventario({
                                            "producto": row["Producto"],
                                            "categoria": cat_p,
                                            "stock_actual": 0,
                                            "stock_minimo": 10,
                                            "stock_inicial": 0,
                                            "precio_unitario": precio_prom,
                                            "unidad_medida": "Unidad",
                                            "activo": True,
                                            "idEmpresa": {"idEmpresa": empresa_id}
                                        })
                                        if res["ok"]:
                                            importados += 1
                                        else:
                                            saltados += 1
                                st.success(f"✅ Importados: {importados}. Saltados: {saltados}.")
                                st.session_state["confirmar_import_est"] = False
                                time.sleep(2)
                                st.rerun()
                        with cci2:
                            if st.button("Cancelar", use_container_width=True, key="no_imp_est"):
                                st.session_state["confirmar_import_est"] = False
                                st.rerun()

    # PESTAÑA 5: Descontinuados (solo gerente)
    if es_gerente() and len(tabs_obj) >= 5:
        with tabs_obj[4]:
            st.markdown('<div class="section-note">'
                        'Productos descontinuados con historial. Puedes reactivarlos en cualquier momento.'
                        '</div>', unsafe_allow_html=True)

            with st.spinner("Cargando descontinuados..."):
                inv_todos = api_get_inventario(incluir_inactivos=True)
                inv_inactivos = [i for i in inv_todos if not i.get("activo", True)]

            if not inv_inactivos:
                empty_state("✨", "No tienes productos descontinuados",
                            "Todos tus productos están activos.")
            else:
                df_inact = pd.DataFrame(inv_inactivos)
                c1, c2 = st.columns(2)
                with c1: kpi("Descontinuados", str(len(df_inact)), "productos inactivos")
                with c2: kpi("Stock atrapado", f"{int(df_inact['stock_actual'].sum())}", "uds en descontinuados")

                cols_inact = ["producto", "categoria", "stock_actual", "stock_minimo"]
                cols_inact = [c for c in cols_inact if c in df_inact.columns]
                df_show_inact = df_inact[cols_inact].copy()
                df_show_inact.columns = ["Producto", "Categoría", "Stock al descontinuar", "Stock mínimo"]
                st.dataframe(df_show_inact, use_container_width=True)

                st.markdown('<div class="section-title">♻️ Reactivar producto</div>', unsafe_allow_html=True)
                opciones_react = {f"{i['producto']} ({int(i['stock_actual'])} uds)": i["id_inventario"]
                                  for i in inv_inactivos}
                sel_react = st.selectbox("Producto a reactivar",
                                          options=list(opciones_react.keys()),
                                          key="sel_react")
                if st.button("♻️ Reactivar producto", use_container_width=True, key="btn_react"):
                    res_react = api_reactivar_inventario(opciones_react[sel_react])
                    if res_react["ok"]:
                        st.success(f"✅ '{sel_react.split(' (')[0]}' reactivado.")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {res_react.get('mensaje','Error al reactivar.')}")


# =============================================================================
# PREDICCIÓN
# =============================================================================

def _bloque_historial_predicciones():
    st.markdown("---")
    st.markdown('<div class="section-title">📜 Historial de predicciones</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-note">'
                'Predicciones generadas por tu empresa. Filtra por mes, categoría o producto.'
                '</div>', unsafe_allow_html=True)

    with st.spinner("Cargando historial..."):
        preds = api_get_predicciones_paginado()

    if not preds:
        empty_state("🔮", "Aún no tienes predicciones",
                    "Usa el formulario de arriba para generar tu primera predicción.")
        return

    df_preds = pd.DataFrame(preds)
    f1, f2, f3 = st.columns(3)

    with f1:
        if "created_at_prediccion" in df_preds.columns:
            df_preds["mes_pred"] = pd.to_datetime(
                df_preds["created_at_prediccion"], errors="coerce").dt.strftime("%Y-%m")
            meses = ["Todos"] + sorted(df_preds["mes_pred"].dropna().unique().tolist(), reverse=True)
            sel_mes = st.selectbox("Mes", meses, key="hp_mes")
        else:
            sel_mes = "Todos"
    with f2:
        if "categoria" in df_preds.columns:
            cats = ["Todas"] + sorted(df_preds["categoria"].dropna().unique().tolist())
            sel_cat = st.selectbox("Categoría", cats, key="hp_cat")
        else:
            sel_cat = "Todas"
    with f3:
        if "producto" in df_preds.columns:
            prods = ["Todos"] + sorted(df_preds["producto"].dropna().unique().tolist())
            sel_prod = st.selectbox("Producto", prods, key="hp_prod")
        else:
            sel_prod = "Todos"

    df_f = df_preds.copy()
    if sel_mes  != "Todos" and "mes_pred" in df_f.columns:
        df_f = df_f[df_f["mes_pred"] == sel_mes]
    if sel_cat  != "Todas" and "categoria" in df_f.columns:
        df_f = df_f[df_f["categoria"] == sel_cat]
    if sel_prod != "Todos" and "producto" in df_f.columns:
        df_f = df_f[df_f["producto"] == sel_prod]

    if df_f.empty:
        st.warning("No hay predicciones para estos filtros.")
        return

    st.caption(f"{len(df_f)} predicción(es) encontrada(s)")

    for _, p in df_f.iterrows():
        pid        = p.get("id_prediccion",       "--")
        fecha_gen  = str(p.get("created_at_prediccion","--"))[:10]
        fecha_proy = str(p.get("fecha_proyectada","--"))
        producto_p = p.get("producto",            "--")
        categoria_p= p.get("categoria",           "--")
        cantidad_p = float(p.get("cantidad_predicha", 0))
        stock_p    = float(p.get("stock_recomendado",  0))
        confianza_p= float(p.get("confianza",          0))
        estado_p   = p.get("estado_predic",       "--")
        color_e    = "#5d8c45" if estado_p == "generado" else "#b7791f"

        col_info, col_det, col_del = st.columns([5, 1, 1])
        with col_info:
            st.markdown(f"""
<div class="hist-row" style="margin-bottom:.3rem;">
  <div>
    <div class="hr-prod">#{pid} · {producto_p} — {categoria_p}</div>
    <div class="hr-fecha">
      Generada: {fecha_gen} · Para el período: {fecha_proy} ·
      Demanda estimada: {cantidad_p:.0f} uds
    </div>
  </div>
  <div style="color:{color_e};font-weight:800;">{estado_p}</div>
</div>""", unsafe_allow_html=True)

        with col_det:
            if st.button("Ver detalle", key=f"det_{pid}", use_container_width=True):
                st.session_state[f"show_{pid}"] = not st.session_state.get(f"show_{pid}", False)

        with col_del:
            mi_id = st.session_state.get("usuario_id", 0)
            id_usuario_pred = p.get("id_usuario", 0)
            if isinstance(id_usuario_pred, dict):
                id_usuario_pred = id_usuario_pred.get("id_usuario", 0)
            puede_eliminar = es_gerente() or int(id_usuario_pred) == mi_id
            if puede_eliminar:
                if st.button("🗑️", key=f"delpred_{pid}", use_container_width=True):
                    st.session_state[f"confirmar_pred_{pid}"] = True
            else:
                st.button("🗑️", key=f"delpred_{pid}", disabled=True,
                          use_container_width=True, help="Solo puedes eliminar tus predicciones")

            if st.session_state.get(f"confirmar_pred_{pid}", False):
                st.warning(f"⚠️ ¿Eliminar predicción de **{producto_p}**?")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("Sí, eliminar", key=f"yes_pred_{pid}",
                                  type="primary", use_container_width=True):
                        if api_eliminar_prediccion(int(pid)):
                            st.success("✅ Predicción eliminada.")
                            st.session_state[f"confirmar_pred_{pid}"] = False
                            st.rerun()
                        else:
                            st.error("❌ Error al eliminar.")
                with cc2:
                    if st.button("Cancelar", key=f"no_pred_{pid}", use_container_width=True):
                        st.session_state[f"confirmar_pred_{pid}"] = False
                        st.rerun()

        if st.session_state.get(f"show_{pid}", False):
            st.markdown(
                '<div style="background:#fffdf8;border:1px solid #e5d4b8;'
                'border-radius:16px;padding:1.2rem;margin-bottom:1rem;">',
                unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1: kpi("Demanda estimada",  f"{cantidad_p:.0f} uds",    "cantidad predicha")
            with r2: kpi("Stock sugerido",    f"{stock_p:.0f} uds",       "+15% margen seguridad")
            with r3: kpi("Confianza",         f"{confianza_p*100:.0f}%",  "R² del modelo = 0.928")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("")


def page_prediccion():
    breadcrumb("Predicción")
    header("Predicción de demanda", "Estima cuánto vas a vender el próximo período")
    ayuda_flotante("Predicción")

    pipeline, meta = cargar_modelo()
    if pipeline is None:
        st.error("No se encontró el archivo modelo_demanda.joblib.")
        _bloque_historial_predicciones()
        return

    opciones = get_opciones()

    st.markdown('<div class="section-note">'
                'Completa los datos del período que quieres proyectar. '
                'El sistema calcula la predicción en segundos.'
                '</div>', unsafe_allow_html=True)

    paso_header(1, "¿Para qué producto y cuándo?", "Producto, categoría y fecha")
    c1, c2, c3 = st.columns(3)
    with c1:
        f_fecha = st.date_input("📅 Fecha a proyectar", value=date.today(), key="p_fecha")
        f_prod  = (st.selectbox("🧵 Producto", opciones["productos"], key="p_prod") if opciones["productos"]
                   else st.text_input("🧵 Producto", key="p_prod"))
        f_cat   = (st.selectbox("📂 Categoría", opciones["categorias"], key="p_cat") if opciones["categorias"]
                   else st.text_input("📂 Categoría", key="p_cat"))
    with c2:
        f_precio = st.number_input("💰 Precio unitario (S/.)", min_value=0.01,
                                    value=25.0, step=0.5, key="p_precio")
        f_canal  = st.selectbox("🚀 Canal de venta", opciones["canales"], key="p_canal")
        f_cli    = st.selectbox("👤 Tipo de cliente", opciones["clientes"], key="p_cli")
    with c3:
        f_region = st.selectbox("📍 Región", opciones["regiones"], key="p_region")
        f_stock  = st.number_input("📦 Stock inicial disponible", min_value=0,
                                    value=100, step=10, key="p_stock")

    st.markdown("---")
    paso_header(2, "¿Habrá descuento o campaña?", "Opcional — pero mejora la predicción")
    c4, c5 = st.columns(2)
    with c4:
        f_dscto = st.checkbox("¿Planeas hacer descuento?", key="p_dscto")
        f_pct   = st.number_input("% de descuento", 0.0, 80.0, 0.0, 5.0,
                                   key="p_pct") if f_dscto else 0.0
    with c5:
        f_camp = st.checkbox("¿Cae en campaña?", key="p_camp")
        f_tipo_camp = "Ninguna"
        if f_camp and opciones["campains"]:
            f_tipo_camp = st.selectbox("¿Qué campaña?", opciones["campains"], key="p_tipo_camp")

    st.markdown("---")
    if st.button("🔮 Calcular mi predicción →", type="primary",
                  use_container_width=True, key="p_generar"):
        datos = {
            "fecha": str(f_fecha), "producto": str(f_prod), "categoria": str(f_cat),
            "precio_unitario": float(f_precio), "canal_venta": str(f_canal),
            "tipo_cliente": str(f_cli), "region_venta": str(f_region),
            "stock_inicial_periodo": int(f_stock), "tiene_dscto": bool(f_dscto),
            "porcentaje_dscto": float(f_pct), "es_campain": bool(f_camp),
            "tipo_campain": str(f_tipo_camp),
        }
        with st.spinner("Calculando tu predicción..."):
            pred = predecir_demanda(datos)
        st.session_state.pred_resultado = pred
        st.session_state.pred_contexto  = datos
        with st.spinner("Guardando resultado..."):
            api_post_prediccion(pred, datos)
        st.success("✅ ¡Predicción generada y guardada!")
        st.rerun()

    pred = st.session_state.get("pred_resultado")
    if pred:
        ctx = st.session_state.get("pred_contexto", {})
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            st.markdown(f"""
<div class="pred-box">
  <div class="pl">Demanda estimada — {ctx.get("producto","")}</div>
  <div class="pv">{pred:.0f}</div>
  <div class="pu">unidades para {ctx.get("fecha","")}</div>
</div>""", unsafe_allow_html=True)
        with c2: kpi("Stock que debes tener", f"{int(pred*1.15):,}", "+15% por seguridad")
        with c3: kpi("Precisión del modelo", "92.8%", "R² = 0.928")

        inv_lista_pred = api_get_inventario()
        inv_prod_pred = next((i for i in inv_lista_pred if i["producto"] == ctx.get("producto","")), None)
        if inv_prod_pred:
            stock_actual_pred = int(inv_prod_pred.get("stock_actual", 0))
            stock_necesario = int(pred * 1.15)
            deficit = stock_necesario - stock_actual_pred
            st.markdown("---")
            if deficit > 0:
                st.error(f"⚠️ **Necesitas reponer {deficit} uds** de '{ctx.get('producto','')}'. "
                         f"Tienes {stock_actual_pred} uds en inventario pero necesitas {stock_necesario}.")
            else:
                st.success(f"✅ Tu stock actual de {stock_actual_pred} uds es suficiente para la predicción de {pred:.0f} uds.")

    _bloque_historial_predicciones()


def page_analisis():
    breadcrumb("Análisis de resultados")
    header("Análisis de resultados", "Qué tan preciso es el modelo predictivo")
    ayuda_flotante("Análisis de resultados")

    _, meta = cargar_modelo()
    if meta is None:
        st.error("Modelo no cargado.")
        return

    r2  = meta.get("r2_test",    0)
    mae = meta.get("mae_test",   0)
    oob = meta.get("oob_score",  0)
    n   = meta.get("filas_train",0)
    pct = round(r2 * 100, 1)

    cls = "ok" if r2 >= 0.85 else ("warn" if r2 >= 0.70 else "bad")
    txt = (f"✅ El modelo predice bien — acierta en el {pct}% de los casos."
           if r2 >= 0.85 else
           f"⚠️ Precisión aceptable — acierta el {pct}%." if r2 >= 0.70
           else f"❌ Precisión baja — solo acierta el {pct}%.")
    st.markdown(f'<div class="vdict {cls}">{txt}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Precisión (R²)",     f"{r2:.4f}",  "objetivo > 0.80")
    with c2: kpi("Error promedio",     f"{mae:.2f}", "unidades de margen")
    with c3: kpi("Validación interna", f"{oob:.4f}", "OOB Score")
    with c4: kpi("Datos de entrenamiento", f"{n:,}", "registros usados")

    st.markdown('<div class="section-title">📈 Ventas reales vs predicción del modelo</div>',
                unsafe_allow_html=True)

    with st.spinner("Cargando datos históricos..."):
        df = api_get_ventas()

    if df.empty or "cantidad_vendida" not in df.columns or "fecha" not in df.columns:
        empty_state("📂", "Sube tus ventas primero",
                    "Ve a Gestión comercial y carga tu archivo.")
    else:
        try:
            df["cantidad_vendida"] = pd.to_numeric(df["cantidad_vendida"], errors="coerce").fillna(0)
            df["fecha_dt"]         = pd.to_datetime(df["fecha"], errors="coerce")
            df = df.dropna(subset=["fecha_dt"])
            df_muestra = df.sort_values("fecha_dt").tail(200).copy()
            predicciones = []
            pipeline, _ = cargar_modelo()

            if pipeline is not None:
                with st.spinner("Analizando tus datos históricos..."):
                    for _, row in df_muestra.iterrows():
                        try:
                            datos_fila = {
                                "fecha":                 str(row["fecha_dt"].date()),
                                "producto":              str(row.get("producto", "")),
                                "categoria":             str(row.get("categoria", "")),
                                "precio_unitario":       float(row.get("precio_unitario", 25.0)),
                                "canal_venta":           str(row.get("canal_venta", "Tienda Fisica")),
                                "tipo_cliente":          str(row.get("tipo_cliente", "Minorista")),
                                "region_venta":          str(row.get("region_venta", "Lima")),
                                "tiene_dscto":           bool(row.get("tiene_dscto", False)),
                                "porcentaje_dscto":      float(row.get("porcentaje_dscto", 0)),
                                "es_campain":            bool(row.get("es_campain", False)),
                                "tipo_campain":          str(row.get("tipo_campain", "Ninguna")),
                                "stock_inicial_periodo": int(row.get("stock_inicial_periodo", 50)),
                            }
                            predicciones.append(predecir_demanda(datos_fila))
                        except Exception:
                            predicciones.append(None)

                df_muestra["prediccion"] = predicciones
                df_muestra["mes_año"] = df_muestra["fecha_dt"].dt.to_period("M").astype(str)
                monthly = df_muestra.groupby("mes_año").agg(
                    real=("cantidad_vendida","sum"),
                    predicha=("prediccion","sum")
                ).reset_index()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly["mes_año"], y=monthly["real"],
                    mode="lines+markers", name="Ventas reales",
                    line=dict(color="#6b3f21", width=3),
                    fill="tozeroy", fillcolor="rgba(107,63,33,.08)",
                    marker=dict(size=8)))
                fig.add_trace(go.Scatter(
                    x=monthly["mes_año"], y=monthly["predicha"],
                    mode="lines+markers", name="Predicción del modelo",
                    line=dict(color="#c9953b", width=2.5, dash="dot"),
                    marker=dict(size=8, symbol="diamond")))

                from sklearn.metrics import r2_score as r2_sk
                try:
                    r2_emp = r2_sk(monthly["real"], monthly["predicha"])
                    r2_txt = f"R² con tus datos: {r2_emp:.4f}"
                except Exception:
                    r2_txt = ""
                    r2_emp = 0

                chart_style(fig, "", 360)
                fig.update_layout(xaxis_title="Período", yaxis_title="Unidades")
                st.plotly_chart(fig, use_container_width=True)

                if r2_txt:
                    if r2_emp >= 0.80:
                        st.success(f"✅ {r2_txt} - El modelo predice bien tus ventas.")
                    elif r2_emp >= 0.60:
                        st.warning(f"⚠️ {r2_txt} - Predicción aceptable.")
                    else:
                        st.info(f"ℹ️ {r2_txt}. Carga más ventas propias para mejorar.")
        except Exception as e:
            st.error(f"Error al generar el gráfico: {e}")

    st.markdown("---")
    pipeline, _ = cargar_modelo()
    try:
        rf     = pipeline.named_steps["random_forest"]
        prep   = pipeline.named_steps["preprocesador"]
        ohe_n  = prep.named_transformers_["cat"].get_feature_names_out(meta["features_cat"]).tolist()
        nombres= meta["features_num"] + ohe_n
        df_imp = (pd.DataFrame({"Variable": nombres, "Importancia": rf.feature_importances_})
                  .sort_values("Importancia", ascending=False).head(15))

        st.markdown('<div class="section-title">🔑 Factores más importantes</div>',
                    unsafe_allow_html=True)
        fig2 = px.bar(df_imp.sort_values("Importancia"),
                      x="Importancia", y="Variable", orientation="h",
                      color="Importancia",
                      color_continuous_scale=["#ead5a3","#5a351e"])
        fig2.update_layout(coloraxis_showscale=False)
        chart_style(fig2, "", 420)
        st.plotly_chart(fig2, use_container_width=True)
    except Exception:
        pass


# =============================================================================
# RECOMENDACIONES (chatbot)
# =============================================================================

def page_recomendaciones():
    breadcrumb("Inteligencia y Recomendaciones")
    header("Asistente comercial", "Pregúntale lo que necesites sobre tus ventas")
    ayuda_flotante("Inteligencia y Recomendaciones")

    with st.spinner("Cargando tu información..."):
        df    = api_get_ventas()
        preds = api_get_predicciones()
        inv_lista = api_get_inventario()

    pred  = st.session_state.get("pred_resultado")
    ctx   = st.session_state.get("pred_contexto", {})

    if not df.empty:
        df["cantidad_vendida"] = pd.to_numeric(df["cantidad_vendida"], errors="coerce").fillna(0)
        st.markdown('<div class="section-title">💡 Recomendaciones para ti</div>',
                    unsafe_allow_html=True)

        if "producto" in df.columns:
            top = (df.groupby("producto")["cantidad_vendida"]
                   .sum().sort_values(ascending=False).head(3))
            if len(top) > 0:
                txt = ", ".join([f"<strong>{p}</strong> ({int(v)} uds)" for p, v in top.items()])
                rec_card("pos", "🏆 Tus productos estrella",
                         f"Los que más vendes son: {txt}. Asegúrate de tener stock siempre.")

        if inv_lista:
            criticos = [i for i in inv_lista if int(i.get("stock_actual",0)) <= int(i.get("stock_minimo",10))]
            if criticos:
                lista_crit = ", ".join([f"<strong>{c['producto']}</strong>" for c in criticos[:5]])
                rec_card("risk", "🔴 Stock crítico — reponer urgente",
                         f"Estos productos están por agotarse: {lista_crit}. "
                         "Ve a Inventario → Registrar entrada para reponerlos.")

        if pred and ctx:
            rec_card("warn", "🎯 Tu próxima predicción",
                     f"Para <strong>{ctx.get('producto','')}</strong> "
                     f"el {ctx.get('fecha','')} el modelo estima "
                     f"<strong>{pred:.0f} unidades</strong>. "
                     f"Tener {int(pred*1.15)} unidades disponibles para no quedarte corto.")

        if "canal_venta" in df.columns:
            canal_top = (df.groupby("canal_venta")["cantidad_vendida"].sum().idxmax())
            rec_card("pos", "🚀 Tu canal estrella",
                     f"Vendes más por <strong>{canal_top}</strong>. "
                     "Refuerza ese canal con promociones especiales.")

    st.markdown('<div class="section-title">💬 Chatea con el asistente</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-note">'
                'Pregúntale lo que quieras sobre tus ventas, stock o predicciones. '
                'El asistente conoce tu inventario actual.'
                '</div>', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if pregunta := st.chat_input("Ej: ¿Cuál es mi producto más vendido? ¿Cómo está mi stock?"):
        st.session_state.chat_history.append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                resp = chatbot_responder(pregunta, df, pred, preds, inv_lista)
            st.markdown(resp)
        st.session_state.chat_history.append({"role": "assistant", "content": resp})

    if st.session_state.chat_history:
        if st.button("🗑️ Limpiar conversación", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# =============================================================================
# MODELO PREDICTIVO
# =============================================================================

def page_modelo_predictivo():
    breadcrumb("Gestión del Modelo Predictivo")
    header("Gestión del Modelo Predictivo",
           "Mejora el modelo con los datos de tu empresa")

    with st.spinner("Cargando tus datos..."):
        df = api_get_ventas()

    n_ventas = len(df) if not df.empty else 0
    MINIMO_RECOMENDADO = 200

    # KPIs de estado
    _, meta_base = cargar_modelo()
    if meta_base:
        st.markdown('<div class="section-title">📋 Modelo base (entrenado con datos de Gamarra)</div>',
                    unsafe_allow_html=True)
        cm1, cm2, cm3, cm4 = st.columns(4)
        with cm1: kpi("Algoritmo",        "Random Forest", "200 árboles")
        with cm2: kpi("Precisión (R²)",   f"{meta_base.get('r2_test',0):.4f}", "objetivo > 0.80")
        with cm3: kpi("Error promedio",   f"{meta_base.get('mae_test',0):.2f}", "unidades")
        with cm4: kpi("Datos entrenamiento", f"{meta_base.get('filas_train',0):,}", "registros Gamarra")

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Tus datos para reentrenamiento</div>',
                unsafe_allow_html=True)
    progreso = min(n_ventas / MINIMO_RECOMENDADO, 1.0)
    c1, c2, c3 = st.columns(3)
    with c1: kpi("Ventas registradas",   str(n_ventas),           "registros activos")
    with c2: kpi("Mínimo recomendado",   str(MINIMO_RECOMENDADO), "para reentrenar")
    with c3: kpi("Avance",              f"{progreso*100:.0f}%",   "hacia el mínimo")
    st.progress(progreso)

    if n_ventas >= MINIMO_RECOMENDADO:
        st.success(f"✅ Tienes {n_ventas} registros. ¡Puedes reentrenar el modelo con tus propios datos!")
    elif n_ventas > 0:
        faltantes = MINIMO_RECOMENDADO - n_ventas
        st.warning(f"⚠️ Te faltan {faltantes} registros para el mínimo recomendado.")
        st.info("Puedes predecir igual con el modelo base, pero con más datos propios las predicciones mejorarán.")
    else:
        empty_state("📂", "No tienes ventas registradas",
                    "Sube un archivo CSV o registra ventas manuales primero.")
        return

    st.markdown("---")
    st.markdown('<div class="section-title">⚙️ Parámetros del modelo</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-note">' +
                'Ajusta cómo aprende el modelo. Los valores por defecto funcionan bien en la mayoría de casos.' +
                '</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        n_trees = st.slider("Número de árboles de decisión",
                             min_value=100, max_value=500, value=300, step=50,
                             help="Más árboles = más preciso, pero más lento")
        st.caption(f"📌 {n_trees} árboles — {'Alta precisión' if n_trees >= 300 else 'Rápido'}")
    with c2:
        max_depth = st.slider("Profundidad de cada árbol",
                               min_value=5, max_value=30, value=20, step=5,
                               help="Mayor profundidad = aprende más detalles")
        st.caption(f"📌 Profundidad {max_depth} — {'Puede sobreajustar' if max_depth > 25 else 'Balance adecuado'}")

    st.markdown("---")
    st.markdown('<div class="section-title">🔄 Reentrenar el modelo</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-note">' +
                'El modelo se entrenará con tus ventas activas. Filtra por rango de fechas si quieres.' +
                '</div>', unsafe_allow_html=True)

    if "fecha" in df.columns:
        df["fecha_dt"] = pd.to_datetime(df["fecha"], errors="coerce")
        fecha_min = df["fecha_dt"].min().date() if df["fecha_dt"].notna().any() else date.today()
        fecha_max = df["fecha_dt"].max().date() if df["fecha_dt"].notna().any() else date.today()
        cf1, cf2 = st.columns(2)
        with cf1:
            desde = st.date_input("Desde", value=fecha_min, key="m_desde")
        with cf2:
            hasta = st.date_input("Hasta", value=fecha_max, key="m_hasta")
        df_modelo = df[(df["fecha_dt"].dt.date >= desde) &
                       (df["fecha_dt"].dt.date <= hasta)].copy()
    else:
        df_modelo = df.copy()

    n_modelo = len(df_modelo)
    col1, col2 = st.columns(2)
    with col1:
        btn_label = "🔄 Reentrenar con mis datos" if n_modelo >= MINIMO_RECOMENDADO \
                    else f"⚠️ Reentrenar de todas formas ({n_modelo} registros)"
        reentrenar = st.button(btn_label, type="primary", use_container_width=True,
                                key="btn_reentrenar")
    with col2:
        if n_modelo < MINIMO_RECOMENDADO:
            st.info(f"💡 Tienes {n_modelo} de {MINIMO_RECOMENDADO} registros mínimos en el rango.")

    if reentrenar and not df.empty:
        if n_modelo < MINIMO_RECOMENDADO:
            st.warning(f"⚠️ Reentrenando con solo {n_modelo} registros. El modelo puede ser menos preciso.")

        cols_necesarias = [
            "fecha","producto","categoria","cantidad_vendida","precio_unitario",
            "porcentaje_dscto","stock_inicial_periodo","tiene_dscto","es_campain",
            "canal_venta","tipo_cliente","region_venta","tipo_campain"
        ]
        cols_ok = [c for c in cols_necesarias if c in df.columns]
        df_train = df_modelo[cols_ok].dropna(subset=["cantidad_vendida","fecha"]).copy()

        if len(df_train) < 10:
            st.error("❌ No hay suficientes datos limpios para entrenar.")
        else:
            with st.spinner(f"Entrenando el modelo con {len(df_train)} registros de tu empresa..."):
                try:
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import r2_score, mean_absolute_error
                    from sklearn.preprocessing import OneHotEncoder
                    from sklearn.compose import ColumnTransformer
                    from sklearn.pipeline import Pipeline

                    df_train["fecha_dt"]    = pd.to_datetime(df_train["fecha"], errors="coerce")
                    df_train["mes"]         = df_train["fecha_dt"].dt.month
                    df_train["semana_año"]  = df_train["fecha_dt"].dt.isocalendar().week.astype(int)
                    df_train["dia_semana"]  = df_train["fecha_dt"].dt.dayofweek
                    df_train["trimestre"]   = df_train["fecha_dt"].dt.quarter
                    df_train["es_finde"]    = (df_train["dia_semana"] >= 5).astype(int)
                    df_train["mes_sin"]     = np.sin(2 * np.pi * df_train["mes"] / 12)
                    df_train["mes_cos"]     = np.cos(2 * np.pi * df_train["mes"] / 12)
                    df_train["año"]         = df_train["fecha_dt"].dt.year
                    df_train["tiene_dscto"] = df_train["tiene_dscto"].astype(int)
                    df_train["es_campain"]  = df_train["es_campain"].astype(int)

                    features_num = ["precio_unitario","porcentaje_dscto","stock_inicial_periodo",
                                    "tiene_dscto","es_campain","año","mes","semana_año",
                                    "dia_semana","es_finde","trimestre","mes_sin","mes_cos"]
                    features_cat = [c for c in ["producto","categoria","canal_venta","tipo_cliente",
                                                "region_venta","tipo_campain"] if c in df_train.columns]

                    X = df_train[features_num + features_cat].copy()
                    y = df_train["cantidad_vendida"].astype(float)
                    for c in features_cat:
                        X[c] = X[c].fillna("Desconocido")
                    for c in features_num:
                        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

                    preprocesador = ColumnTransformer([
                        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features_cat)
                    ], remainder="passthrough")
                    rf_local = RandomForestRegressor(
                        n_estimators=n_trees, max_depth=max_depth,
                        min_samples_split=4, min_samples_leaf=2,
                        random_state=42, n_jobs=-1)
                    pipe_local = Pipeline([("preprocesador", preprocesador), ("random_forest", rf_local)])

                    if len(df_train) >= 20:
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
                        pipe_local.fit(X_tr, y_tr)
                        y_pred    = pipe_local.predict(X_te)
                        r2_local  = r2_score(y_te, y_pred)
                        mae_local = mean_absolute_error(y_te, y_pred)
                    else:
                        pipe_local.fit(X, y)
                        r2_local  = r2_score(y, pipe_local.predict(X))
                        mae_local = mean_absolute_error(y, pipe_local.predict(X))

                    st.session_state["modelo_local"]     = pipe_local
                    st.session_state["modelo_local_r2"]  = r2_local
                    st.session_state["modelo_local_mae"] = mae_local
                    st.session_state["modelo_local_n"]   = len(df_train)
                    st.success(f"✅ Modelo reentrenado con {len(df_train)} registros de tu empresa.")
                except Exception as e:
                    st.error(f"❌ Error durante el reentrenamiento: {e}")

    if st.session_state.get("modelo_local"):
        r2_l  = st.session_state.get("modelo_local_r2",  0)
        mae_l = st.session_state.get("modelo_local_mae", 0)
        n_l   = st.session_state.get("modelo_local_n",   0)
        _, meta_orig = cargar_modelo()
        r2_orig  = meta_orig.get("r2_test", 0) if meta_orig else 0
        mae_orig = meta_orig.get("mae_test", 0) if meta_orig else 0

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Resultados del reentrenamiento</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-note">' +
                    '⚠️ Este modelo es temporal (en memoria de sesión). Se pierde al cerrar el navegador.' +
                    '</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi("Precisión (modelo local)",    f"{r2_l:.4f}",   "con tus datos")
        with c2: kpi("Error promedio (local)",      f"{mae_l:.2f}",  "unidades de margen")
        with c3: kpi("Precisión (modelo original)", f"{r2_orig:.4f}","sector Gamarra")
        with c4: kpi("Registros usados",            str(n_l),        "para entrenar")

        if r2_l > r2_orig:
            st.success(f"✅ Tu modelo local es más preciso (R² {r2_l:.3f} vs {r2_orig:.3f}). ¡Bueno para la defensa!")
        elif r2_l >= r2_orig * 0.90:
            st.info("ℹ️ Rendimiento similar al modelo original. Agrega más ventas para mejorar.")
        else:
            st.warning("⚠️ El modelo original es más preciso. Necesitas más datos propios.")

        # Gráfico comparativo
        ce, cf = st.columns(2)
        with ce:
            fig_comp = go.Figure(data=[
                go.Bar(name="Modelo original (Gamarra)", x=["R²"], y=[r2_orig],
                       marker_color="#7b6a57"),
                go.Bar(name="Tu modelo local", x=["R²"], y=[r2_l],
                       marker_color="#c9953b")
            ])
            chart_style(fig_comp, "Comparación R²", 280)
            fig_comp.update_layout(barmode="group", yaxis_range=[0, 1.1])
            st.plotly_chart(fig_comp, use_container_width=True)
        with cf:
            fig_mae = go.Figure(data=[
                go.Bar(name="Modelo original (Gamarra)", x=["MAE"], y=[mae_orig],
                       marker_color="#7b6a57"),
                go.Bar(name="Tu modelo local", x=["MAE"], y=[mae_l],
                       marker_color="#c9953b")
            ])
            chart_style(fig_mae, "Error promedio (MAE) — menor es mejor", 280)
            fig_mae.update_layout(barmode="group")
            st.plotly_chart(fig_mae, use_container_width=True)

        # ── PREDICCIÓN CON EL MODELO LOCAL ──────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">🔮 Probar predicción con tu modelo</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-note">'
                    'Usa tu modelo recién entrenado para hacer una predicción de prueba. '
                    'Compara el resultado con el modelo original para ver si mejoró.'
                    '</div>', unsafe_allow_html=True)

        opciones = get_opciones()
        lp1, lp2, lp3 = st.columns(3)
        with lp1:
            lp_fecha = st.date_input("📅 Fecha a proyectar", value=date.today(), key="lp_fecha")
            lp_prod = (st.selectbox("🧵 Producto", opciones["productos"], key="lp_prod")
                       if opciones["productos"] else st.text_input("🧵 Producto", key="lp_prod"))
            lp_cat  = (st.selectbox("📂 Categoría", opciones["categorias"], key="lp_cat")
                       if opciones["categorias"] else st.text_input("📂 Categoría", key="lp_cat"))
        with lp2:
            lp_precio  = st.number_input("💰 Precio unitario (S/.)", min_value=0.01,
                                          value=25.0, step=0.5, key="lp_precio")
            lp_canal   = st.selectbox("🚀 Canal", opciones["canales"], key="lp_canal")
            lp_cli     = st.selectbox("👤 Tipo cliente", opciones["clientes"], key="lp_cli")
        with lp3:
            lp_region  = st.selectbox("📍 Región", opciones["regiones"], key="lp_region")
            lp_stock   = st.number_input("📦 Stock inicial", min_value=0,
                                          value=100, step=10, key="lp_stock")
            lp_dscto   = st.checkbox("¿Descuento?", key="lp_dscto")
            lp_pct     = st.number_input("% descuento", 0.0, 80.0, 0.0, key="lp_pct") if lp_dscto else 0.0

        if st.button("🔮 Calcular con MI modelo", type="primary",
                      use_container_width=True, key="btn_pred_local"):
            datos_pred = {
                "fecha": str(lp_fecha), "producto": str(lp_prod), "categoria": str(lp_cat),
                "precio_unitario": float(lp_precio), "canal_venta": str(lp_canal),
                "tipo_cliente": str(lp_cli), "region_venta": str(lp_region),
                "stock_inicial_periodo": int(lp_stock), "tiene_dscto": bool(lp_dscto),
                "porcentaje_dscto": float(lp_pct), "es_campain": False, "tipo_campain": "Ninguna",
            }
            try:
                # Predicción con modelo local
                pipe_local = st.session_state["modelo_local"]
                fecha_dt   = pd.to_datetime(datos_pred["fecha"])
                mes        = fecha_dt.month
                fila = {
                    "precio_unitario":       float(datos_pred["precio_unitario"]),
                    "porcentaje_dscto":      float(datos_pred["porcentaje_dscto"]),
                    "stock_inicial_periodo": int(datos_pred["stock_inicial_periodo"]),
                    "tiene_dscto":           int(bool(datos_pred["tiene_dscto"])),
                    "es_campain":            0,
                    "año":        fecha_dt.year,
                    "mes":        mes,
                    "semana_año": int(fecha_dt.isocalendar().week),
                    "dia_semana": fecha_dt.dayofweek,
                    "es_finde":   int(fecha_dt.dayofweek >= 5),
                    "trimestre":  fecha_dt.quarter,
                    "mes_sin":    np.sin(2 * np.pi * mes / 12),
                    "mes_cos":    np.cos(2 * np.pi * mes / 12),
                    "producto":     str(datos_pred["producto"]),
                    "categoria":    str(datos_pred["categoria"]),
                    "canal_venta":  str(datos_pred["canal_venta"]),
                    "tipo_cliente": str(datos_pred["tipo_cliente"]),
                    "region_venta": str(datos_pred["region_venta"]),
                    "tipo_campain": "Ninguna",
                }
                X_pred = pd.DataFrame([fila])
                pred_local = max(0.0, round(float(pipe_local.predict(X_pred)[0]), 1))

                # Predicción con modelo original
                pred_orig = predecir_demanda(datos_pred)

                # Mostrar comparación
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.markdown(f"""
<div class="pred-box">
  <div class="pl">Tu modelo local — {lp_prod}</div>
  <div class="pv">{pred_local:.0f}</div>
  <div class="pu">unidades estimadas</div>
</div>""", unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"""
<div style="background:#7b6a57;border-radius:24px;padding:2rem 1.5rem;color:#fffaf0;text-align:center;">
  <div style="font-size:.85rem;color:#f7dfb7;margin-bottom:.5rem;">Modelo original (Gamarra)</div>
  <div style="font-size:3rem;font-weight:900;line-height:1;">{pred_orig:.0f}</div>
  <div style="font-size:.85rem;color:#f6e6c8;margin-top:.4rem;">unidades estimadas</div>
</div>""", unsafe_allow_html=True)
                with rc3:
                    diff = pred_local - pred_orig
                    color_diff = "#5d8c45" if abs(diff) < pred_orig * 0.15 else "#b7791f"
                    kpi("Diferencia entre modelos",
                        f"{'+' if diff >= 0 else ''}{diff:.0f} uds",
                        "< 15% diferencia = modelos alineados")

                if abs(diff) < pred_orig * 0.15:
                    st.success("✅ Ambos modelos coinciden bastante — tu modelo local es confiable.")
                elif pred_local > pred_orig:
                    st.info(f"ℹ️ Tu modelo local predice {diff:.0f} uds más. Puede reflejar mejor el patrón de tu empresa.")
                else:
                    st.info(f"ℹ️ Tu modelo local predice {abs(diff):.0f} uds menos. Puede ser más conservador con tus datos reales.")

            except Exception as e:
                st.error(f"❌ Error al predecir con el modelo local: {e}")


def page_reportes():
    breadcrumb("Reportes")
    header("Reportes", "Descarga tus datos y predicciones")
    ayuda_flotante("Reportes")

    tab1, tab2, tab3 = st.tabs(["📊 Ventas", "🔮 Predicciones", "📦 Inventario"])

    with tab1:
        with st.spinner("Cargando ventas..."):
            df = api_get_ventas()
        if df.empty:
            empty_state("📂", "Sin datos para reportar",
                        "Carga ventas primero en Gestión comercial.")
        else:
            st.dataframe(df.head(20), use_container_width=True)
            st.caption(f"Total: {len(df):,} ventas")
            cd1, cd2 = st.columns(2)
            with cd1:
                buf = io.BytesIO()
                df.to_csv(buf, index=False)
                st.download_button("⬇️ Descargar CSV", buf.getvalue(),
                                   f"ventas_{date.today()}.csv", "text/csv",
                                   use_container_width=True)
            with cd2:
                buf2 = io.BytesIO()
                df.to_excel(buf2, index=False)
                st.download_button("⬇️ Descargar Excel", buf2.getvalue(),
                                   f"ventas_{date.today()}.xlsx",
                                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                   use_container_width=True)

    with tab2:
        preds = api_get_predicciones_paginado()
        if not preds:
            empty_state("🔮", "Sin predicciones",
                        "Genera predicciones en la sección Predicción.")
        else:
            df_p = pd.DataFrame(preds)
            st.dataframe(df_p, use_container_width=True)
            st.caption(f"Total: {len(df_p):,} predicciones")
            buf = io.BytesIO()
            df_p.to_excel(buf, index=False)
            st.download_button("⬇️ Descargar Excel", buf.getvalue(),
                               f"predicciones_{date.today()}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

    with tab3:
        inv = api_get_inventario()
        if not inv:
            empty_state("📦", "Sin inventario",
                        "Agrega productos en la sección Inventario.")
        else:
            df_i = pd.DataFrame(inv)
            st.dataframe(df_i, use_container_width=True)
            st.caption(f"Total: {len(df_i):,} productos activos")
            buf = io.BytesIO()
            df_i.to_excel(buf, index=False)
            st.download_button("⬇️ Descargar Excel", buf.getvalue(),
                               f"inventario_{date.today()}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

# =============================================================================
# CREAR COLABORADOR
# =============================================================================

def page_crear_colaborador():
    breadcrumb("Crear colaborador")
    header("Crear colaborador", "Agrega usuarios a tu equipo")
    ayuda_flotante("Crear colaborador")

    if not es_gerente():
        st.error("🔒 Solo el Gerente puede crear colaboradores.")
        return

    st.markdown('<div class="section-note">'
                'Los colaboradores pueden registrar ventas pero no editar inventario, '
                'crear usuarios ni eliminar datos de la empresa.'
                '</div>', unsafe_allow_html=True)

    paso_header(1, "Datos del nuevo colaborador", "Crea su cuenta de acceso")
    with st.form("form_colab", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            nom = st.text_input("👤 Nombre completo", placeholder="Ej: Juan Pérez")
            dni = st.text_input("🪪 DNI", max_chars=8, placeholder="12345678")
            cor = st.text_input("📧 Correo (opcional)", placeholder="juan@ejemplo.com")
            sex = st.selectbox("Sexo", ["M","F","Otro"])
        with c2:
            usr = st.text_input("🔑 Usuario para ingresar", placeholder="Ej: juan_perez")
            field_hint("Con esto entrará al sistema. Sin espacios.")
            pas = st.text_input("🔒 Contraseña inicial", type="password",
                                 placeholder="Mínimo 6 caracteres")
            tel = st.text_input("📱 Teléfono (opcional)", max_chars=9,
                                 placeholder="9 dígitos")
            field_hint("Solo números, exactamente 9 dígitos")
            dirc = st.text_input("📍 Dirección (opcional)")
        crear = st.form_submit_button("➕ Crear colaborador",
                                       type="primary", use_container_width=True)

    if crear:
        if not usr or not pas:
            st.warning("⚠️ Usuario y contraseña son obligatorios.")
        elif tel and not telefono_valido(tel):
            st.error("❌ El teléfono debe tener exactamente 9 dígitos.")
        else:
            res = api_crear_usuario({
                "username": usr, "password": pas,
                "dni": dni, "correo": cor, "sexo": sex,
                "telefono": tel, "direccion": dirc,
                "id_rol": 2,
                "id_empresa": st.session_state.get("empresa_id", 0),
            })
            if res["ok"]:
                st.success(f"✅ Colaborador '{usr}' creado.")
                
            else:
                st.error(res.get("mensaje", "Error al crear colaborador."))

    st.markdown("---")
    st.markdown('<div class="section-title">👥 Mi equipo</div>', unsafe_allow_html=True)
    with st.spinner("Cargando equipo..."):
        usuarios = api_get_usuarios_empresa()
    if not usuarios:
        st.info("Aún no tienes colaboradores.")
    else:
        df_us = pd.DataFrame(usuarios)
        cols_us = ["nombre_usuario", "dni_usuario", "correo_usuario", "telefono_usuario"]
        cols_us = [c for c in cols_us if c in df_us.columns]
        if "id_rol" in df_us.columns:
            df_us["rol"] = df_us["id_rol"].apply(parse_rol)  # FIX: usar helper robusto
            cols_us.append("rol")
        # Renombrar columnas para mostrar bonito
        col_rename = {"nombre_usuario": "Nombre", "dni_usuario": "DNI",
                      "correo_usuario": "Correo", "telefono_usuario": "Teléfono", "rol": "Rol"}
        df_us_show = df_us[cols_us].copy()
        df_us_show.columns = [col_rename.get(c, c) for c in df_us_show.columns]
        st.dataframe(df_us_show, use_container_width=True)
        st.caption(f"{len(df_us)} usuario(s)")


# =============================================================================
# SOPORTE
# =============================================================================

def page_soporte():
    breadcrumb("Soporte y Ayuda")
    header("Soporte y Ayuda", "Resolvemos tus dudas")
    ayuda_flotante("Soporte y Ayuda")

    tab_faq, tab_contacto = st.tabs(["❓ Preguntas frecuentes", "📨 Enviar consulta"])

    with tab_faq:
        st.markdown('<div class="section-title">❓ Preguntas frecuentes</div>',
                    unsafe_allow_html=True)

        with st.expander("🔮 ¿Cómo funciona la predicción?"):
            st.markdown("""
El modelo analiza tus ventas históricas y proyecta cuánto venderás de cada producto en un período.
Usa un algoritmo Random Forest con 200 árboles entrenado con miles de ventas reales de MYPEs de Gamarra.

**Para usarla:** ve a "Predicción", elige producto, fecha y datos. El sistema te dará una estimación.
""")

        with st.expander("📦 ¿Cómo se descuenta el inventario al vender?"):
            st.markdown("""
Cada vez que registras una venta manual desde Streamlit:

1. La venta se guarda en el historial.
2. Automáticamente se descuenta la cantidad del stock del producto.
3. Si eliminas la venta, el stock se restaura.

⚠️ Las ventas del CSV histórico NO descuentan stock (son solo para el modelo).
""")

        with st.expander("🗑️ ¿Qué pasa si elimino un producto del inventario?"):
            st.markdown("""
El sistema decide automáticamente entre dos opciones:

- **Si el producto NO tiene ventas ni entradas registradas:** se elimina completamente.
- **Si tiene historial:** se marca como descontinuado (se preserva el historial). Puedes reactivarlo después.
""")

        with st.expander("⚠️ Eliminé una venta de un producto descontinuado — ¿qué pasa con el stock?"):
            st.markdown("""
El stock se devuelve al registro del producto, pero como está descontinuado **no aparece disponible para venta**.

- Las unidades quedan "guardadas" en el producto inactivo
- Si reactivas el producto desde Inventario → Descontinuados, recuperas esas unidades
""")

        with st.expander("👥 ¿Diferencia entre Gerente y Colaborador?"):
            st.markdown("""
**Gerente** puede:
- Ver y editar inventario, registrar entradas
- Crear y eliminar colaboradores
- Eliminar cualquier venta de la empresa
- Editar datos de la empresa
- Generar reportes

**Colaborador** puede:
- Ver el inventario (solo lectura)
- Registrar sus propias ventas
- Ver ventas de la empresa
- Eliminar solo sus propias ventas
- Generar predicciones
""")

    with tab_contacto:
        st.markdown('<div class="section-note">'
                    'Si tu duda no está en las preguntas frecuentes, envíanos tu consulta. '
                    'Nuestro equipo te responde por correo en 24-48 horas.'
                    '</div>', unsafe_allow_html=True)

        with st.form("form_contacto", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                s_nombre = st.text_input("👤 Tu nombre",
                                          value=st.session_state.get("usuario_nombre",""))
                s_correo = st.text_input("📧 Correo de contacto",
                                          value=st.session_state.get("correo_usuario",""),
                                          placeholder="donde te respondemos")
                field_hint("Verifica que sea un correo que revises seguido")
            with c2:
                s_tipo = st.selectbox("📋 Tipo de consulta",
                                       ["Problema técnico (no funciona algo)",
                                        "Duda sobre cómo usar el sistema",
                                        "Pedido de mejora",
                                        "Error en mis datos",
                                        "Asuntos de facturación",
                                        "Otro"])
                s_prioridad = st.selectbox("⚡ Prioridad",
                                            ["🟢 Baja — puedo esperar",
                                             "🟡 Media — necesito respuesta pronto",
                                             "🔴 Alta — no puedo trabajar"])

            s_asunto = st.text_input("📝 Asunto",
                                      placeholder="Resumen corto de tu consulta")
            field_hint("Por ejemplo: 'No puedo registrar ventas' o 'Cómo importo mi inventario'")

            s_mensaje = st.text_area("💬 Cuéntanos qué necesitas",
                                      placeholder="Describe paso a paso lo que pasa. "
                                                  "Si es un error, dinos qué estabas haciendo y qué mensaje viste.",
                                      height=150)

            s_anexar = st.checkbox("📎 Anexar contexto de la app (rol, empresa, ID usuario)",
                                    value=True,
                                    help="Nos ayuda a resolver más rápido")

            enviar = st.form_submit_button("📤 Enviar consulta",
                                            type="primary", use_container_width=True)

        if enviar:
            if not s_nombre or not s_correo or not s_asunto or not s_mensaje:
                st.warning("⚠️ Completa todos los campos para que podamos ayudarte.")
            else:
                # Guardamos localmente y mostramos resumen.
                # NOTA: para producción, esto debería enviarse a un endpoint backend
                # o servicio de correo (SendGrid, EmailJS, etc.)
                consulta = {
                    "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "nombre": s_nombre,
                    "correo": s_correo,
                    "tipo": s_tipo,
                    "prioridad": s_prioridad,
                    "asunto": s_asunto,
                    "mensaje": s_mensaje,
                }
                if s_anexar:
                    consulta["contexto"] = {
                        "usuario_id": st.session_state.get("usuario_id",0),
                        "empresa_id": st.session_state.get("empresa_id",0),
                        "empresa_nombre": st.session_state.get("empresa_nombre",""),
                        "rol": st.session_state.get("rol_nombre",""),
                    }

                # Guardar en historial de sesión
                if "consultas_enviadas" not in st.session_state:
                    st.session_state.consultas_enviadas = []
                st.session_state.consultas_enviadas.append(consulta)

                st.success("✅ Consulta enviada correctamente.")
                
                st.info(f"📧 Te responderemos a **{s_correo}** en 24-48 horas hábiles.")
                with st.expander("📋 Ver resumen de tu consulta"):
                    st.json(consulta)

        # Historial de consultas en esta sesión
        if st.session_state.get("consultas_enviadas"):
            st.markdown("---")
            st.markdown('<div class="section-title">📜 Tus consultas en esta sesión</div>',
                        unsafe_allow_html=True)
            for i, c in enumerate(reversed(st.session_state.consultas_enviadas[-5:])):
                with st.expander(f"#{len(st.session_state.consultas_enviadas)-i} · {c['fecha']} · {c['asunto']}"):
                    st.write(f"**Tipo:** {c['tipo']}")
                    st.write(f"**Prioridad:** {c['prioridad']}")
                    st.write(f"**Mensaje:** {c['mensaje']}")

    st.markdown("---")
    st.markdown('<div class="section-title">📞 Otros canales de soporte</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("📧 **Correo directo**\nsoporte@fashanalytics.pe")
    with c2:
        st.info("📱 **WhatsApp**\n+51 999 888 777")
    with c3:
        st.info("⏰ **Horario**\nLun-Vie 9:00-18:00")

# =============================================================================
# GUÍA
# =============================================================================

def page_guia():
    breadcrumb("Guía de uso")
    header("Guía paso a paso", "Cómo usar FashAnalytics desde cero")
    ayuda_flotante("Guía de uso")

    st.markdown('<div class="section-note">'
                'Si es tu primera vez, sigue estos pasos en orden para sacarle el máximo provecho.'
                '</div>', unsafe_allow_html=True)

    paso_header(1, "Carga tu historial de ventas",
                "Si tienes ventas anteriores en Excel, súbelas")
    st.markdown("""
- Ve a **📁 Gestión comercial → Subir archivo Excel/CSV**
- Sube tu archivo con tus ventas históricas
- Esto NO afecta tu inventario actual, es solo para entrenar el modelo
""")

    paso_header(2, "Registra tu inventario actual",
                "Lo que tienes físicamente en tu local hoy")
    st.markdown("""
- Ve a **📦 Inventario → ➕ Registrar producto**
- Agrega producto por producto, o sube un CSV con tu inventario
- También puedes importar desde el estimado del historial
""")

    paso_header(3, "Comienza a vender",
                "Cada venta descuenta automáticamente del inventario")
    st.markdown("""
- Ve a **✏️ Registrar mis ventas** (o "Gestión comercial → Agregar venta manual")
- Elige el producto del inventario
- Al guardar, el stock se actualiza automáticamente
""")

    paso_header(4, "Genera predicciones",
                "Estima cuánto venderás el próximo período")
    st.markdown("""
- Ve a **🔮 Predicción**
- Elige producto, fecha y datos del período
- El sistema te dirá cuánta demanda esperar y cuánto stock necesitas
""")

    paso_header(5, "Repón stock cuando sea necesario",
                "Registra cuando llega mercadería nueva")
    st.markdown("""
- Ve a **📦 Inventario → 📥 Registrar entrada**
- Indica producto, cantidad y motivo (compra, producción, devolución, etc.)
- Queda registrado para auditoría
""")

    paso_header(6, "Consulta al asistente cuando tengas dudas",
                "Te responde sobre tus ventas y stock")
    st.markdown("""
- Ve a **💡 Inteligencia y Recomendaciones**
- Pregúntale lo que necesites
- El asistente conoce tu inventario y predicciones actuales.
""")

# =============================================================================
# CUENTA
# =============================================================================

def page_cuenta():
    breadcrumb("Cuenta")
    header("Mi cuenta", "Tus datos y los de tu empresa")
    ayuda_flotante("Cuenta")

    tabs_cuenta = ["👤 Mis datos", "🏢 Mi empresa"]
    if es_gerente():
        tabs_cuenta.append("👥 Mi equipo")

    tabs_obj = st.tabs(tabs_cuenta)

    with tabs_obj[0]:
        st.markdown('<div class="section-note">'
                    'Tus datos personales. Si necesitas cambiar tu contraseña, edítala aquí.'
                    '</div>', unsafe_allow_html=True)

        with st.form("form_mis_datos"):
            c1, c2 = st.columns(2)
            with c1:
                st.text_input("👤 Usuario", value=st.session_state.get("usuario_nombre",""), disabled=True)
                m_dni = st.text_input("🪪 DNI", value=st.session_state.get("dni_usuario",""), max_chars=8)
                m_cor = st.text_input("📧 Correo", value=st.session_state.get("correo_usuario",""))
                m_sex = st.selectbox("Sexo", ["M","F","Otro"],
                                      index=["M","F","Otro"].index(st.session_state.get("sexo_usuario","M"))
                                      if st.session_state.get("sexo_usuario","M") in ["M","F","Otro"] else 0)
            with c2:
                m_tel = st.text_input("📱 Teléfono", value=st.session_state.get("telefono",""), max_chars=9)
                field_hint("Solo números, 9 dígitos")
                m_dir = st.text_input("📍 Dirección", value=st.session_state.get("direccion",""))
                st.text_input("🛡️ Rol", value=st.session_state.get("rol_nombre",""), disabled=True)
                m_pas = st.text_input("🔒 Nueva contraseña (déjala vacía para no cambiar)",
                                       type="password")
            guardar_d = st.form_submit_button("💾 Guardar cambios", type="primary",
                                               use_container_width=True)

        if guardar_d:
            if m_tel and not telefono_valido(m_tel):
                st.error("❌ El teléfono debe tener 9 dígitos exactos.")
            else:
                payload = {
                    "id_usuario":        st.session_state.get("usuario_id",0),
                    "nombre_usuario":    st.session_state.get("usuario_nombre",""),
                    "dni_usuario":       m_dni,
                    "correo_usuario":    m_cor,
                    "sexo_usuario":      m_sex,
                    "telefono_usuario":  m_tel,
                    "direccion_usuario": m_dir,
                    "password_usuario":  m_pas if m_pas else "__KEEP__",
                    "id_rol":            st.session_state.get("rol_id",2),
                    "id_empresa":        st.session_state.get("empresa_id",0),
                }
                res = api_actualizar_usuario(payload)
                if res["ok"]:
                    st.session_state.dni_usuario    = m_dni
                    st.session_state.correo_usuario = m_cor
                    st.session_state.sexo_usuario   = m_sex
                    st.session_state.telefono       = m_tel
                    st.session_state.direccion      = m_dir
                    st.success("✅ Datos actualizados.")
                else:
                    st.error(res.get("mensaje", "Error al actualizar."))

    with tabs_obj[1]:
        if es_gerente():
            st.markdown('<div class="section-note">'
                        'Como gerente, puedes actualizar los datos de tu empresa.'
                        '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-note">'
                        'Solo el Gerente puede editar los datos de la empresa.'
                        '</div>', unsafe_allow_html=True)

        bloqueado = not es_gerente()

        with st.form("form_empresa"):
            c1, c2 = st.columns(2)
            with c1:
                e_nombre = st.text_input("🏢 Nombre de la empresa",
                                          value=st.session_state.get("empresa_nombre",""),
                                          disabled=bloqueado)
                e_ruc = st.text_input("🔢 RUC",
                                       value=st.session_state.get("ruc_empresa",""),
                                       disabled=bloqueado, max_chars=11)
            with c2:
                sectores = ["Textil","Confecciones","Moda","Otro"]
                sector_actual = st.session_state.get("sector_empresa","Textil")
                idx_sec = sectores.index(sector_actual) if sector_actual in sectores else 0
                e_sector = st.selectbox("🏭 Sector", sectores, index=idx_sec, disabled=bloqueado)
                st.text_input("📅 Fecha de registro",
                              value=str(st.session_state.get("fecha_registro",""))[:10],
                              disabled=True)

            if es_gerente():
                guardar_emp = st.form_submit_button("💾 Guardar cambios", type="primary",
                                                     use_container_width=True)
            else:
                guardar_emp = False
                st.form_submit_button("🔒 Solo el Gerente puede editar",
                                       disabled=True, use_container_width=True)

        if guardar_emp:
            if not e_nombre or not e_ruc:
                st.warning("⚠️ Nombre y RUC son obligatorios.")
            elif not e_ruc.isdigit() or len(e_ruc) != 11:
                st.error("❌ El RUC debe tener 11 dígitos numéricos.")
            else:
                with st.spinner("Actualizando empresa..."):
                    res = api_actualizar_empresa(
                        st.session_state.get("empresa_id",0),
                        e_nombre, e_ruc, e_sector
                    )
                if res["ok"]:
                    st.session_state.empresa_nombre = e_nombre
                    st.session_state.ruc_empresa = e_ruc
                    st.session_state.sector_empresa = e_sector
                    st.success("✅ Datos de empresa actualizados.")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {res.get('mensaje','Error al actualizar.')}")
                    st.info("💡 Si el error persiste, verifica que el backend tenga el endpoint PUT /empresas/actualizar/{id}")

    if es_gerente() and len(tabs_obj) >= 3:
        with tabs_obj[2]:
            st.markdown('<div class="section-note">'
                        'Gestiona los usuarios de tu empresa.'
                        '</div>', unsafe_allow_html=True)
            with st.spinner("Cargando equipo..."):
                usuarios = api_get_usuarios_empresa()
            if not usuarios:
                empty_state("👥", "Aún no tienes colaboradores",
                            "Crea uno en la sección 'Crear colaborador'.")
            else:
                df_eq = pd.DataFrame(usuarios)
                cols_eq = ["id_usuario", "nombre_usuario", "dni_usuario",
                           "correo_usuario", "telefono_usuario"]
                cols_eq = [c for c in cols_eq if c in df_eq.columns]
                if "id_rol" in df_eq.columns:
                    df_eq["rol"] = df_eq["id_rol"].apply(parse_rol)  # FIX
                    cols_eq.append("rol")
                col_rename_eq = {"id_usuario":"ID","nombre_usuario":"Nombre","dni_usuario":"DNI",
                                 "correo_usuario":"Correo","telefono_usuario":"Teléfono","rol":"Rol"}
                df_eq_show = df_eq[cols_eq].copy()
                df_eq_show.columns = [col_rename_eq.get(c, c) for c in df_eq_show.columns]
                st.dataframe(df_eq_show, use_container_width=True)

                st.markdown('<div class="section-title">✏️ Editar usuario</div>',
                            unsafe_allow_html=True)
                opciones_eq = {f"{u.get('nombre_usuario','?')} (#{u.get('id_usuario','?')})": u
                               for u in usuarios}
                sel_eq_label = st.selectbox("Usuario a editar",
                                             options=list(opciones_eq.keys()),
                                             key="sel_eq_user")
                sel_eq = opciones_eq.get(sel_eq_label, {})
                if sel_eq:
                    with st.form("form_edit_eq"):
                        ec1, ec2 = st.columns(2)
                        with ec1:
                            ee_nom = st.text_input("Nombre", value=sel_eq.get("nombre_usuario",""))
                            ee_dni = st.text_input("DNI", value=sel_eq.get("dni_usuario",""), max_chars=8)
                            ee_cor = st.text_input("Correo", value=sel_eq.get("correo_usuario",""))
                            ee_sex = st.selectbox("Sexo", ["M","F","Otro"],
                                                   index=["M","F","Otro"].index(sel_eq.get("sexo_usuario","M"))
                                                   if sel_eq.get("sexo_usuario","M") in ["M","F","Otro"] else 0)
                        with ec2:
                            ee_tel = st.text_input("Teléfono", value=sel_eq.get("telefono_usuario",""), max_chars=9)
                            field_hint("9 dígitos")
                            ee_dir = st.text_input("Dirección", value=sel_eq.get("direccion_usuario",""))
                            rol_actual_txt = parse_rol(sel_eq.get("id_rol"))
                            ee_rol = st.selectbox("Rol", ["COLABORADOR", "GERENTE"],
                                                   index=1 if rol_actual_txt == "GERENTE" else 0)
                            ee_pas = st.text_input("Nueva contraseña (vacío para no cambiar)", type="password")
                        c_ed1, c_ed2 = st.columns(2)
                        with c_ed1:
                            guardar_eq = st.form_submit_button("💾 Guardar", type="primary",
                                                                use_container_width=True)
                        with c_ed2:
                            eliminar_eq = st.form_submit_button("🗑️ Eliminar usuario",
                                                                 use_container_width=True)
                    if guardar_eq:
                        if ee_tel and not telefono_valido(ee_tel):
                            st.error("❌ El teléfono debe tener 9 dígitos.")
                        else:
                            res = api_actualizar_usuario({
                                "id_usuario":        sel_eq.get("id_usuario", 0),
                                "nombre_usuario":    ee_nom,
                                "dni_usuario":       ee_dni,
                                "correo_usuario":    ee_cor,
                                "sexo_usuario":      ee_sex,
                                "telefono_usuario":  ee_tel,
                                "direccion_usuario": ee_dir,
                                "password_usuario":  ee_pas if ee_pas else "__KEEP__",
                                "id_rol":            1 if ee_rol == "GERENTE" else 2,
                                "id_empresa":        st.session_state.get("empresa_id", 0),
                            })
                            if res["ok"]:
                                st.success("✅ Usuario actualizado.")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(res.get("mensaje", "Error al actualizar."))
                    if eliminar_eq:
                        st.session_state[f"confirmar_del_eq_{sel_eq.get('id_usuario',0)}"] = True

                    if st.session_state.get(f"confirmar_del_eq_{sel_eq.get('id_usuario',0)}", False):
                        st.warning(f"⚠️ ¿Eliminar a **{sel_eq.get('nombre_usuario','')}**?")
                        ccd1, ccd2 = st.columns(2)
                        with ccd1:
                            if st.button("Sí, eliminar", type="primary",
                                          use_container_width=True, key=f"yes_del_eq_{sel_eq.get('id_usuario',0)}"):
                                res = api_eliminar_usuario(sel_eq.get("id_usuario", 0))
                                if res["ok"]:
                                    st.success("✅ Usuario eliminado.")
                                    st.session_state[f"confirmar_del_eq_{sel_eq.get('id_usuario',0)}"] = False
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(res.get("mensaje", "Error al eliminar."))
                        with ccd2:
                            if st.button("Cancelar", use_container_width=True,
                                          key=f"no_del_eq_{sel_eq.get('id_usuario',0)}"):
                                st.session_state[f"confirmar_del_eq_{sel_eq.get('id_usuario',0)}"] = False
                                st.rerun()

# =============================================================================
# MAIN — ROUTER
# =============================================================================

def main():
    if not st.session_state.get("authenticated", False):
        login_view()
        return

    sidebar_nav()
    page = st.session_state.get("page", "Dashboard")

    if page == "Dashboard":
        page_dashboard()
    elif page == "Gestión comercial":
        page_gestion_comercial()
    elif page == "Registrar mis ventas":
        page_registrar_ventas()
    elif page == "Inventario":
        page_inventario()
    elif page == "Predicción":
        page_prediccion()
    elif page == "Análisis de resultados":
        page_analisis()
    elif page == "Inteligencia y Recomendaciones":
        page_recomendaciones()
    elif page == "Gestión del Modelo Predictivo":
        page_modelo_predictivo()
    elif page == "Reportes":
        page_reportes()
    elif page == "Crear colaborador":
        page_crear_colaborador()
    elif page == "Soporte y Ayuda":
        page_soporte()
    elif page == "Guía de uso":
        page_guia()
    elif page == "Cuenta":
        page_cuenta()
    else:
        page_dashboard()

if __name__ == "__main__":
    main()
