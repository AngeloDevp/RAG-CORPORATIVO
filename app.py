import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuración visual
st.set_page_config(page_title="Asistente RAG", page_icon="🏢")
st.title("🏢 Asistente Corporativo Privado")
st.caption("Conectado a servidor IA local (ASUS G14) vía Red de Alta Velocidad")

# 1. INICIALIZACIÓN GLOBAL DE MODELOS (Caché para velocidad)
@st.cache_resource
def cargar_modelos():
    # ATENCIÓN: Actualiza esta IP si cambió al conectar el cable de red
    ip_ia = "http://192.168.137.1:11434" 
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ip_ia)
    llm = OllamaLLM(model="phi3", base_url=ip_ia)
    return embeddings, llm

embeddings, llm = cargar_modelos()

# 2. FUNCIÓN PARA PROCESAR EL PDF DINÁMICO
def procesar_pdf(archivo_subido):
    # Guardar el archivo en una ruta temporal de Linux
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(archivo_subido.getvalue())
        tmp_ruta = tmp_file.name

    # Cargar y extraer el texto del PDF
    loader = PyPDFLoader(tmp_ruta)
    documentos = loader.load()
    
    # Dividir el texto en fragmentos digeribles para la IA
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    textos_divididos = splitter.split_documents(documentos)

    # Crear una nueva base de datos vectorial para este documento específico
    vectorstore = Chroma.from_documents(
        documents=textos_divididos,
        embedding=embeddings,
        persist_directory="./db_dinamica" # Usamos otra carpeta para no mezclar
    )
    
    # Borrar el archivo temporal por limpieza
    os.remove(tmp_ruta)
    
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. INTERFAZ DE SUBIDA (BARRA LATERAL)
with st.sidebar:
    st.header("📄 Gestión de Documentos")
    archivo_pdf = st.file_uploader("Sube un documento PDF confidencial", type=["pdf"])
    
    # Lógica de procesamiento
    if archivo_pdf is not None:
        # Verificamos si es un archivo nuevo para no reprocesarlo en cada clic
        if "nombre_archivo" not in st.session_state or st.session_state.nombre_archivo != archivo_pdf.name:
            with st.spinner("Leyendo y vectorizando documento..."):
                # Procesamos el PDF y guardamos el "buscador" (retriever) en memoria
                st.session_state.retriever = procesar_pdf(archivo_pdf)
                st.session_state.nombre_archivo = archivo_pdf.name
                st.session_state.mensajes = [] # Limpiamos el chat al subir un nuevo PDF
                st.success("¡Documento procesado y listo!")

# 4. CONSTRUCCIÓN DEL CHAT (Solo si ya se subió un PDF)
if "retriever" in st.session_state:
    
    template = """Eres un asistente corporativo experto. Usa exclusivamente los siguientes fragmentos de contexto para responder a la pregunta de forma clara y concisa.
    Si la respuesta a la pregunta no se encuentra en el contexto proporcionado, responde exactamente "No tengo informacion sobre esto en el documento proporcionado". No inventes datos.

    Contexto recuperado:
    {context}

    pregunta del usuario: {question}

    respuesta:"""
    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Armamos la cadena usando el retriever guardado en la memoria de la sesión
    rag_chain = (
        {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Mostrar mensajes anteriores
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["rol"]):
            st.markdown(mensaje["contenido"])

    # Interacción del usuario
    if pregunta_usuario := st.chat_input("Haz una pregunta sobre el PDF subido..."):
        
        st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_usuario})
        with st.chat_message("user"):
            st.markdown(pregunta_usuario)

        with st.chat_message("assistant"):
            with st.spinner("Analizando información..."):
                try:
                    respuesta = rag_chain.invoke(pregunta_usuario)
                    st.markdown(respuesta)
                    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})
                except Exception as e:
                    st.error(f"Error de ejecución: {e}")
else:
    # Mensaje de bienvenida si no hay archivos
    st.info("👈 Por favor, sube un documento PDF en el panel lateral izquierdo para comenzar el análisis.")