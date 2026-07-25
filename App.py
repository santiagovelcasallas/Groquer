"""
Dashboard de Entrevistas -> Tabla -> EDA
========================================
Flujo:
  1) Defines qué campos quieres extraer de cada entrevista.
  2) Ingresas cada entrevista pegando texto O subiendo un audio
     (Groq Whisper lo transcribe con la MISMA API key).
  3) Un LLM de Groq (Llama) convierte cada texto en una fila estructurada.
  4) Se acumulan todas las entrevistas en un DataFrame.
  5) EDA estadístico con pandas + numpy.

Todo determinista: temperature=0 y salida forzada a JSON.

Requisitos:
    pip install streamlit groq pandas numpy plotly

Ejecutar:
    streamlit run app_entrevistas.py
"""

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from groq import Groq

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Entrevistas → EDA", page_icon="🎙️", layout="wide")
st.title("🎙️ Entrevistas → Tabla → EDA")

# ---------------------------------------------------------------------------
# Estado de la sesión
# ---------------------------------------------------------------------------
if "filas" not in st.session_state:
    st.session_state.filas = []          # lista de dicts, cada uno = una entrevista
if "esquema" not in st.session_state:
    # Esquema por defecto (edítalo en la pestaña "1. Campos")
    st.session_state.esquema = pd.DataFrame(
        [
            {"campo": "edad",           "tipo": "numerico",   "descripcion": "Edad de la persona en años"},
            {"campo": "genero",         "tipo": "categorico", "descripcion": "Género: masculino / femenino / otro"},
            {"campo": "satisfaccion",   "tipo": "numerico",   "descripcion": "Nivel de satisfacción del 1 al 10"},
            {"campo": "sentimiento",    "tipo": "categorico", "descripcion": "positivo / neutro / negativo"},
            {"campo": "tema_principal", "tipo": "categorico", "descripcion": "Tema central que menciona"},
            {"campo": "recomendaria",   "tipo": "categorico", "descripcion": "si / no"},
            {"campo": "resumen",        "tipo": "texto",      "descripcion": "Resumen en una frase"},
        ]
    )

# ---------------------------------------------------------------------------
# Sidebar: credenciales y modelos
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")
    api_key = st.text_input("Groq API Key", type="password",
                            help="La misma key sirve para Llama y para Whisper.")
    modelo_llm = st.selectbox(
        "Modelo de extracción (LLM)",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        help="70b = mejor calidad extrayendo. 8b = más barato pero menos preciso.",
    )
    modelo_audio = st.selectbox(
        "Modelo de transcripción (audio)",
        ["whisper-large-v3-turbo", "whisper-large-v3"],
        help="turbo = más rápido y barato.",
    )
    idioma = st.text_input("Idioma del audio (ISO)", value="es")
    st.divider()
    st.caption("Extracción y transcripción con temperature=0 → resultados reproducibles.")


def get_client():
    if not api_key:
        st.error("Introduce tu Groq API Key en la barra lateral.")
        st.stop()
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Funciones núcleo
# ---------------------------------------------------------------------------
def transcribir_audio(client, archivo) -> str:
    """Audio -> texto usando Groq Whisper (determinista)."""
    datos = archivo.read()
    resp = client.audio.transcriptions.create(
        file=(archivo.name, datos),
        model=modelo_audio,
        language=idioma or None,
        temperature=0,
        response_format="text",
    )
    # Con response_format="text" la respuesta es directamente el string
    return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))


def extraer_estructura(client, texto: str, esquema: pd.DataFrame) -> dict:
    """Texto de entrevista -> dict con los campos del esquema (JSON determinista)."""
    # Construimos la descripción del esquema para el prompt
    lineas = [
        f'- "{r.campo}" ({r.tipo}): {r.descripcion}'
        for r in esquema.itertuples()
    ]
    esquema_txt = "\n".join(lineas)
    campos = list(esquema["campo"])

    system = (
        "Eres un extractor de datos de entrevistas. Devuelves EXCLUSIVAMENTE un "
        "objeto JSON válido con exactamente estas claves y nada más. "
        "Si un dato no aparece en el texto, usa null. "
        "Los campos 'numerico' deben ser números (no texto). "
        "Los campos 'categorico' deben usar categorías cortas y consistentes "
        "(minúsculas, sin explicación)."
    )
    user = (
        f"Campos a extraer:\n{esquema_txt}\n\n"
        f"Devuelve un JSON con estas claves exactas: {campos}\n\n"
        f"Entrevista:\n\"\"\"\n{texto}\n\"\"\""
    )

    resp = client.chat.completions.create(
        model=modelo_llm,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    contenido = resp.choices[0].message.content
    try:
        data = json.loads(contenido)
    except json.JSONDecodeError:
        data = {}
    # Nos aseguramos de que estén todas las claves del esquema
    return {c: data.get(c, None) for c in campos}


def construir_df() -> pd.DataFrame:
    """Convierte las filas acumuladas en un DataFrame tipado según el esquema."""
    df = pd.DataFrame(st.session_state.filas)
    if df.empty:
        return df
    for r in st.session_state.esquema.itertuples():
        if r.campo in df.columns and r.tipo == "numerico":
            df[r.campo] = pd.to_numeric(df[r.campo], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
tab_campos, tab_ingesta, tab_tabla, tab_eda = st.tabs(
    ["1. Campos", "2. Ingesta", "3. Tabla", "4. EDA"]
)

# ---- 1. CAMPOS -------------------------------------------------------------
with tab_campos:
    st.subheader("Define qué extraer de cada entrevista")
    st.caption("Añade, borra o edita filas. 'tipo' controla cómo se analiza en el EDA.")
    editado = st.data_editor(
        st.session_state.esquema,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "tipo": st.column_config.SelectboxColumn(
                options=["numerico", "categorico", "texto"]
            )
        },
        key="editor_esquema",
    )
    st.session_state.esquema = editado.dropna(subset=["campo"]).reset_index(drop=True)

# ---- 2. INGESTA ------------------------------------------------------------
with tab_ingesta:
    st.subheader("Añade una entrevista")
    modo = st.radio("Fuente", ["Pegar texto", "Subir audio"], horizontal=True)

    texto_entrevista = ""
    if modo == "Pegar texto":
        texto_entrevista = st.text_area("Texto de la entrevista", height=220)
    else:
        audio = st.file_uploader(
            "Archivo de audio", type=["mp3", "wav", "m4a", "ogg", "flac", "webm", "mp4"]
        )
        if audio and st.button("🔊 Transcribir"):
            with st.spinner("Transcribiendo con Whisper..."):
                client = get_client()
                texto_entrevista = transcribir_audio(client, audio)
                st.session_state.transcripcion = texto_entrevista
        texto_entrevista = st.text_area(
            "Transcripción (puedes corregirla antes de extraer)",
            value=st.session_state.get("transcripcion", ""),
            height=220,
        )

    if st.button("➕ Extraer y añadir a la tabla", type="primary"):
        if not texto_entrevista.strip():
            st.warning("No hay texto que procesar.")
        else:
            with st.spinner("Extrayendo con el LLM..."):
                client = get_client()
                fila = extraer_estructura(client, texto_entrevista, st.session_state.esquema)
                fila["_texto"] = texto_entrevista  # guardamos el origen por trazabilidad
                st.session_state.filas.append(fila)
            st.success(f"Añadida. Total de entrevistas: {len(st.session_state.filas)}")
            st.json(fila)

# ---- 3. TABLA --------------------------------------------------------------
with tab_tabla:
    st.subheader("Datos estructurados")
    df = construir_df()
    if df.empty:
        st.info("Aún no hay entrevistas. Añade alguna en la pestaña 'Ingesta'.")
    else:
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ Descargar CSV",
                df.to_csv(index=False).encode("utf-8"),
                "entrevistas.csv",
                "text/csv",
            )
        with c2:
            if st.button("🗑️ Vaciar todo"):
                st.session_state.filas = []
                st.rerun()

# ---- 4. EDA ----------------------------------------------------------------
with tab_eda:
    st.subheader("Análisis exploratorio (pandas + numpy)")
    df = construir_df()
    if df.empty:
        st.info("Necesitas datos para el EDA.")
    else:
        # Columnas por tipo según el esquema
        num_cols = [r.campo for r in st.session_state.esquema.itertuples()
                    if r.tipo == "numerico" and r.campo in df.columns]
        cat_cols = [r.campo for r in st.session_state.esquema.itertuples()
                    if r.tipo == "categorico" and r.campo in df.columns]

        # --- Resumen general ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Entrevistas", len(df))
        m2.metric("Variables numéricas", len(num_cols))
        m3.metric("Variables categóricas", len(cat_cols))

        # --- Valores faltantes ---
        st.markdown("**Valores faltantes por columna**")
        faltantes = df.isna().sum()
        st.dataframe(
            faltantes[faltantes > 0].rename("nulos").to_frame()
            if faltantes.sum() else pd.DataFrame({"info": ["Sin valores faltantes"]}),
            use_container_width=True,
        )

        # --- Estadística descriptiva numérica ---
        if num_cols:
            st.markdown("**Estadística descriptiva (numéricas)**")
            desc = df[num_cols].describe().T
            desc["mediana"] = df[num_cols].median()
            desc["cv"] = np.where(desc["mean"] != 0, desc["std"] / desc["mean"], np.nan)
            st.dataframe(desc, use_container_width=True)

            col = st.selectbox("Histograma de:", num_cols)
            st.plotly_chart(px.histogram(df, x=col, nbins=20, marginal="box"),
                            use_container_width=True)

            # --- Correlaciones (numpy vía df.corr) ---
            if len(num_cols) >= 2:
                st.markdown("**Matriz de correlación**")
                corr = df[num_cols].corr(numeric_only=True)
                st.plotly_chart(
                    px.imshow(corr, text_auto=".2f", aspect="auto",
                              color_continuous_scale="RdBu", zmin=-1, zmax=1),
                    use_container_width=True,
                )

        # --- Categóricas ---
        if cat_cols:
            st.markdown("**Frecuencias (categóricas)**")
            col_cat = st.selectbox("Variable categórica:", cat_cols)
            conteo = df[col_cat].value_counts(dropna=False).reset_index()
            conteo.columns = [col_cat, "frecuencia"]
            cc1, cc2 = st.columns([2, 1])
            with cc1:
                st.plotly_chart(px.bar(conteo, x=col_cat, y="frecuencia"),
                                use_container_width=True)
            with cc2:
                st.dataframe(conteo, use_container_width=True)

            # Cruce categórica vs numérica
            if num_cols:
                st.markdown("**Media de una numérica por categoría**")
                cnum = st.selectbox("Numérica:", num_cols, key="cruce_num")
                agg = df.groupby(col_cat)[cnum].agg(["mean", "median", "count"])
                st.dataframe(agg, use_container_width=True)
