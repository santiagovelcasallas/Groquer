# 🎙️ Dashboard de Entrevistas → Tabla → EDA

Dashboard en Streamlit que toma entrevistas (texto pegado o audio), las convierte
en filas estructuradas con un LLM de **Groq** (Llama) y ejecuta un análisis
exploratorio (EDA) con **pandas** y **numpy**.

El audio se transcribe con **Whisper de Groq**, usando la misma API key: no hace
falta ningún otro proveedor.

---

## Flujo

1. **Campos** — defines qué extraer de cada entrevista (edad, satisfacción,
   sentimiento, tema…) y marcas cada campo como `numerico`, `categorico` o `texto`.
2. **Ingesta** — por cada persona pegas el texto o subes el audio. Si subes audio,
   Whisper lo transcribe y puedes corregir la transcripción antes de extraer.
   El LLM devuelve una fila en JSON y la acumula.
3. **Tabla** — todas las entrevistas juntas en un DataFrame, descargable a CSV.
4. **EDA** — descriptivos, valores faltantes, histogramas, matriz de correlación,
   frecuencias de categóricas y cruces categórica × numérica.

Todo es **determinista**: extracción con `temperature=0` y salida forzada a JSON,
así la misma entrevista produce siempre la misma fila.

---

## Instalación

Necesitas Python 3.9 o superior.

```bash
# 1. Clona o descarga esta carpeta y entra en ella
cd dashboard-entrevistas

# 2. (Recomendado) crea un entorno virtual
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

# 3. Instala dependencias
pip install -r requirements.txt
```

---

## Ejecución

```bash
streamlit run app.py
```

Se abrirá en el navegador (por defecto http://localhost:8501).

Pega tu **Groq API Key** en la barra lateral. Consíguela gratis en
https://console.groq.com/keys

---

## Modelos usados

| Uso            | Modelo por defecto          | Coste aprox.            |
|----------------|-----------------------------|-------------------------|
| Extracción     | `llama-3.3-70b-versatile`   | $0.59 / $0.79 por 1M tokens |
| Extracción (económico) | `llama-3.1-8b-instant` | $0.05 / $0.08 por 1M tokens |
| Transcripción  | `whisper-large-v3-turbo`    | ~$0.04 por hora de audio |

El modelo se elige desde la barra lateral. Para estructurar entrevistas se
recomienda el `70b`; el `8b` es más barato pero se equivoca más extrayendo campos.

---

## Estructura del proyecto

```
dashboard-entrevistas/
├── app.py             # Aplicación Streamlit
├── requirements.txt   # Dependencias
└── README.md          # Este archivo
```

---

## Notas

- **Categorías consistentes**: el prompt pide categorías cortas en minúsculas, pero
  si necesitas que "positivo" no aparezca a veces como "muy positivo", conviene
  añadir un paso de normalización posterior.
- **Carga en lote**: la app procesa una entrevista por vez. Para muchas entrevistas
  puedes extender la pestaña de Ingesta para subir varios archivos y recorrerlos en bucle.
- **Privacidad**: el texto de las entrevistas se envía a la API de Groq para su
  procesamiento. Ten esto en cuenta si manejas datos sensibles.
