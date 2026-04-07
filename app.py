# app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

# ---------------------------
# 🔹 Configuración
# ---------------------------
load_dotenv()
API_URL = os.getenv("FASTAPI_URL", "http://192.168.137.1:8000")

st.set_page_config(page_title="Asistente Híbrido", page_icon="🤖", layout="wide")
st.title("🤖 Mi IA Corporativa")

# ---------------------------
# 1️⃣ Sidebar para PDFs (opcional)
# ---------------------------
with st.sidebar:
    st.header("📁 Documentos (Opcional)")
    archivo_pdf = st.file_uploader("Sube un PDF para darle contexto a la IA", type=["pdf"])

    if archivo_pdf and st.button("Procesar Archivo"):
        with st.spinner("Procesando PDF..."):
            files = {"file": (archivo_pdf.name, archivo_pdf.getvalue(), "application/pdf")}
            try:
                r = requests.post(f"{API_URL}/upload", files=files)
                if r.status_code == 200:
                    st.success("PDF cargado y procesado correctamente.")
                else:
                    st.error(f"Error {r.status_code}: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con la API.")

# ---------------------------
# 2️⃣ Inicializar historial interno
# ---------------------------
if "historial" not in st.session_state:
    st.session_state.historial = []

# ---------------------------
# 3️⃣ Mostrar chat existente
# ---------------------------
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["rol"]):
        st.markdown(mensaje["contenido"])

# ---------------------------
# 4️⃣ Entrada de usuario
# ---------------------------
if prompt := st.chat_input("Escribe algo..."):
    # Mostrar pregunta inmediatamente
    st.chat_message("user").markdown(prompt)
    st.session_state.historial.append({"rol": "user", "contenido": prompt})

    # Placeholder para respuesta del modelo
    mensaje_assistant = st.chat_message("assistant")
    mensaje_assistant.markdown("💬 Pensando...")

    payload = {
        "pregunta": prompt,
        "historial": st.session_state.historial
    }

    try:
        res = requests.post(f"{API_URL}/chat", json=payload)
        if res.status_code == 200:
            respuesta = res.json().get("respuesta", "No recibí respuesta válida.")
            
            # Mostrar respuesta final directamente
            mensaje_assistant.markdown(respuesta)
            st.session_state.historial.append({"rol": "assistant", "contenido": respuesta})
        else:
            mensaje_assistant.markdown(f"❌ Error {res.status_code}: {res.text}")
    except requests.exceptions.ConnectionError:
        mensaje_assistant.markdown("❌ No se pudo conectar con la API.")