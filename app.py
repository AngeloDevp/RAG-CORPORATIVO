import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Configuración visual
st.set_page_config(page_title="Asistente RAG", page_icon="🏢")
st.title("🏢 Asistente Corporativo Privado")
st.caption("Conectado a servidor IA local (ASUS G14) vía Red de Alta Velocidad")

# 1. INICIALIZACIÓN GLOBAL DE MODELOS (Caché para velocidad)
@st.cache_resource
def cargar_modelos():
    # ATENCIÓN: Asegúrate de que esta IP sea la correcta de tu ASUS
    ip_ia = "http://192.168.137.1:11434" 
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ip_ia)
    llm = OllamaLLM(model="phi3", base_url=ip_ia)
    return embeddings, llm

embeddings, llm = cargar_modelos()

# 2. FUNCIÓN PARA PROCESAR EL PDF DINÁMICO
def procesar_pdf(archivo_subido):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(archivo_subido.getvalue())
        tmp_ruta = tmp_file.name

    loader = PyPDFLoader(tmp_ruta)
    documentos = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos_divididos = splitter.split_documents(documentos)

    vectorstore = Chroma.from_documents(
        documents=textos_divididos,
        embedding=embeddings,
        persist_directory="./db_dinamica"
    )
    
    os.remove(tmp_ruta)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. INTERFAZ DE SUBIDA (BARRA LATERAL)
with st.sidebar:
    st.header("📄 Gestión de Documentos")
    archivo_pdf = st.file_uploader("Sube un documento PDF confidencial", type=["pdf"])
    
    if archivo_pdf is not None:
        if "nombre_archivo" not in st.session_state or st.session_state.nombre_archivo != archivo_pdf.name:
            with st.spinner("Leyendo y vectorizando documento..."):
                st.session_state.retriever = procesar_pdf(archivo_pdf)
                st.session_state.nombre_archivo = archivo_pdf.name
                st.session_state.mensajes = [] # Limpiamos el chat al subir un nuevo PDF
                st.success("¡Documento procesado y listo!")

# 4. CONSTRUCCIÓN DEL CHAT CON MEMORIA CONVERSACIONAL
if "retriever" in st.session_state:
    
    # 🌟 NUEVO TEMPLATE: Ahora incluye una variable para el historial
    template = """Eres un asistente corporativo experto. Usa exclusivamente los siguientes fragmentos de contexto y el historial de la conversación para responder a la pregunta de forma clara y concisa.
    Si la respuesta a la pregunta no se encuentra en el contexto proporcionado, responde exactamente "No tengo informacion sobre esto en el documento proporcionado". No inventes datos.

    Historial reciente de la conversación:
    {historial}

    Contexto recuperado del documento:
    {context}

    Pregunta actual del usuario: {question}

    Respuesta:"""
    
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    def format_historial(mensajes):
        # 🌟 LOGICA DE MEMORIA: Tomamos solo los últimos 4 mensajes (2 interacciones) 
        # para no saturar la ventana de contexto del modelo Phi3
        ultimos_mensajes = mensajes[-4:] 
        historial_formateado = []
        for m in ultimos_mensajes:
            rol = "Usuario" if m["rol"] == "user" else "Asistente"
            historial_formateado.append(f"{rol}: {m['contenido']}")
        
        if not historial_formateado:
            return "No hay historial previo. Esta es la primera pregunta."
        return "\n".join(historial_formateado)

    # Cadena simplificada para inyectar variables manualmente
    rag_chain = prompt | llm | StrOutputParser()

    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    if pregunta_usuario := st.chat_input("Haz una pregunta sobre el PDF subido..."):
        
        # 1. Guardar y mostrar la pregunta del usuario
        st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_usuario})
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

        with st.chat_message("assistant"):
            with st.spinner("Analizando historial y buscando en el documento..."):
                try:
                    # 2. Recuperar documentos de la base de datos
                    documentos = st.session_state.retriever.invoke(pregunta_usuario)
                    contexto_str = format_docs(documentos)
                    
                    # 3. Formatear el historial (excluyendo la pregunta actual que ya se guardó)
                    historial_str = format_historial(st.session_state.mensajes[:-1])
                    
                    # 4. Inyectar todo al modelo
                    respuesta = rag_chain.invoke({
                        "context": contexto_str,
                        "historial": historial_str,
                        "question": pregunta_usuario
                    })
                    
                    st.markdown(respuesta)
                    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
                except Exception as e:
                    st.error(f"Error de ejecución: {e}")
else:
    st.info("👈 Por favor, sube un documento PDF en el panel lateral izquierdo para comenzar el análisis.")