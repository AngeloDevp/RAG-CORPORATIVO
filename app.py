import streamlit as st
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()
# Asegúrate de que esta sea la IP de tu laptop ASUS
API_URL = os.getenv("FASTAPI_URL", "http://192.168.137.2:8000") 

st.set_page_config(page_title="Asistente Híbrido", page_icon="🤖")
st.title("🤖 Mi IA Corporativa")

# --- 1. Sidebar para archivos ---
with st.sidebar:
    st.header("📁 Documentos (Opcional)")
    archivo_pdf = st.file_uploader("Sube un PDF para darle contexto a la IA", type=["pdf"])
    
    if archivo_pdf and st.button("Procesar Archivo"):
        with st.spinner("Enviando a la ASUS y vectorizando..."):
            files = {"file": (archivo_pdf.name, archivo_pdf.getvalue(), "application/pdf")}
            try:
                r = requests.post(f"{API_URL}/upload", files=files)
                if r.status_code == 200:
                    st.success("PDF cargado y procesado exitosamente.")
                    st.session_state.contexto_activo = True
                else:
                    st.error(f"Error {r.status_code} al subir el archivo.")
            except requests.exceptions.ConnectionError:
                st.error("Error de conexión. Verifica que la ASUS esté encendida y corriendo la API.")

# --- 2. Historial de Chat ---
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Mostrar mensajes anteriores
for m in st.session_state.mensajes:
    with st.chat_message(m["rol"]): 
        st.markdown(m["contenido"])

# --- 3. Generador para el Efecto Máquina de Escribir ---
def stream_local(texto):
    """Toma un texto completo y lo 'escupe' palabra por palabra."""
    for palabra in texto.split(" "):
        yield palabra + " "
        time.sleep(0.04) # Ajusta este número (ej. 0.02) para que escriba más rápido o más lento

# --- 4. Entrada del Usuario ---
if pregunta := st.chat_input("Escribe algo..."):
    # Guardar y mostrar la pregunta del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user"): 
        st.markdown(pregunta)

    # Procesar la respuesta de la IA
    with st.chat_message("assistant"):
        # Armamos el JSON exactamente como la API lo espera (soluciona el Error 422)
        payload = {
            "pregunta": pregunta, 
            "historial": st.session_state.mensajes[:-1]
        }
        
        try:
            # Enviamos la petición sin 'stream=True' porque la API devuelve un JSON
            res = requests.post(f"{API_URL}/chat", json=payload)
            
            if res.status_code == 200:
                # 1. Extraemos el texto de la respuesta JSON
                datos = res.json()
                texto_respuesta = datos.get("respuesta", "Lo siento, no recibí una respuesta válida.")
                
                # 2. Imprimimos el texto usando nuestro generador para el efecto visual
                texto_ia = st.write_stream(stream_local(texto_respuesta))
                
                # 3. Guardamos en el historial
                st.session_state.mensajes.append({"rol": "assistant", "contenido": texto_ia})
            else:
                st.error(f"Error {res.status_code}: {res.text}")
                
        except requests.exceptions.ConnectionError:
            st.error("Error: No me pude comunicar con el servidor en la ASUS.")