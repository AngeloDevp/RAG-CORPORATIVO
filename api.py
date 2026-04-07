from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from fastapi.responses import StreamingResponse

load_dotenv()  # Carga variables de entorno desde .env

# Inicializamos la API
app = FastAPI(
    title="API RAG Corporativo", 
    description="Backend para procesamiento de IA Local y Base de Datos Vectorial",
    version="1.0.0"
)

# 1. CONFIGURACIÓN GLOBAL
# ATENCIÓN: Verifica que esta sea la IP de tu ASUS por cable
IP_IA = os.getenv("OLLAMA_URL", "http:////127.0.0.1:11434")

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=IP_IA)
llm = OllamaLLM(model="phi3", base_url=IP_IA)

# Variable global para mantener la base de datos cargada en memoria
vectorstore_retriever = None

# 2. MODELOS DE DATOS (Pydantic valida que los datos entren correctamente)
class Mensaje(BaseModel):
    rol: str
    contenido: str

class ChatRequest(BaseModel):
    pregunta: str
    historial: List[Mensaje] = []

# 3. ENDPOINT: SUBIDA DE DOCUMENTOS
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore_retriever
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF")
        
    # Guardamos el archivo recibido temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(await file.read())
        tmp_ruta = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_ruta)
        documentos = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        textos_divididos = splitter.split_documents(documentos)

        vectorstore = Chroma.from_documents(
            documents=textos_divididos,
            embedding=embeddings,
            persist_directory="./db_api" # Nueva carpeta para la API
        )
        
        vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        return {"mensaje": "Documento procesado y vectorizado exitosamente", "archivo": file.filename}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando documento: {str(e)}")
    finally:
        os.remove(tmp_ruta) # Limpieza

# 4. ENDPOINT: CHAT E IA (AHORA CON STREAMING)
@app.post("/chat")
async def chat(request: ChatRequest):
    global vectorstore_retriever
    
    if not vectorstore_retriever:
        raise HTTPException(status_code=400, detail="Debes subir un documento primero en /upload")

    template = """Eres un asistente corporativo experto. Usa exclusivamente los siguientes fragmentos de contexto y el historial de la conversación para responder a la pregunta de forma clara y concisa.
    Si la respuesta a la pregunta no se encuentra en el contexto proporcionado, responde exactamente "No tengo informacion sobre esto en el documento proporcionado". No inventes datos.

    Historial reciente de la conversación:
    {historial}

    Contexto recuperado del documento:
    {context}

    Pregunta actual del usuario: {question}

    Respuesta:"""
    
    prompt = PromptTemplate.from_template(template)
    rag_chain = prompt | llm | StrOutputParser()

    # Recuperamos fragmentos de la BD Vectorial
    documentos = vectorstore_retriever.invoke(request.pregunta)
    contexto_str = "\n\n".join(doc.page_content for doc in documentos)

    # Formateamos el historial recibido
    if request.historial:
        ultimos_mensajes = request.historial[-4:]
        lineas_historial = [f"{'Usuario' if m.rol == 'user' else 'Asistente'}: {m.contenido}" for m in ultimos_mensajes]
        historial_str = "\n".join(lineas_historial)
    else:
        historial_str = "No hay historial previo."

    # Función generadora que envía los pedazos de texto al instante
    async def generar_respuesta():
        try:
            # En lugar de .invoke(), usamos .stream()
            for chunk in rag_chain.stream({
                "context": contexto_str,
                "historial": historial_str,
                "question": request.pregunta
            }):
                yield chunk # 'yield' es lo que hace la magia de enviar pedazo a pedazo
        except Exception as e:
            yield f"\n\n[Error de transmisión: {str(e)}]"

    # Retornamos la respuesta como un flujo de texto continuo
    return StreamingResponse(generar_respuesta(), media_type="text/plain")