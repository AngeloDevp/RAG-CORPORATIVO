import os
import tempfile
import shutil
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

app = FastAPI(
    title="API RAG Corporativo",
    version="1.0.0",
    description="Backend para IA local y base de datos vectorial RAG"
)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
llm = OllamaLLM(model="phi3", base_url=OLLAMA_URL)

# Variables globales
vectorstore_retriever = None
vectorstore_instance = None
current_db_folder = None

# Pydantic
class Mensaje(BaseModel):
    rol: str
    contenido: str

class ChatRequest(BaseModel):
    pregunta: str
    historial: List[Mensaje] = []

# ---------------------------
# Endpoint: Upload PDF
# ---------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore_retriever, vectorstore_instance, current_db_folder

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo PDFs permitidos")

    if vectorstore_instance:
        try:
            vectorstore_instance.close()
        except Exception:
            pass
        vectorstore_instance = None
        vectorstore_retriever = None

    if current_db_folder and os.path.exists(current_db_folder):
        shutil.rmtree(current_db_folder, ignore_errors=True)

    current_db_folder = os.path.join("db_api", str(uuid.uuid4()))
    os.makedirs(current_db_folder, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        ruta_tmp = tmp.name

    try:
        loader = PyPDFLoader(ruta_tmp)
        documentos = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documentos)

        vectorstore_instance = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=current_db_folder
        )
        vectorstore_retriever = vectorstore_instance.as_retriever(search_kwargs={"k": 3})

        return {"mensaje": "Documento cargado correctamente", "db_folder": current_db_folder}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando PDF: {str(e)}")
    finally:
        if os.path.exists(ruta_tmp):
            os.remove(ruta_tmp)

# ---------------------------
# Endpoint: Chat
# ---------------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    global vectorstore_retriever

    # Limitar historial a últimas 3 interacciones
    historial_corto = request.historial[-3:] if request.historial else []

    historial_str = "\n".join(
        f"{'Usuario' if m.rol == 'user' else 'Asistente'}: {m.contenido}"
        for m in historial_corto
    ) or "Sin historial previo."

    try:
        if vectorstore_retriever:
            # RAG → usar contexto de PDF
            documentos = vectorstore_retriever.invoke(request.pregunta)
            contexto_str = "\n\n".join(doc.page_content for doc in documentos)

            template = """
Usa SOLO la información del contexto para responder de manera clara y concisa.
Si la respuesta no se encuentra en el contexto, responde:
"No tengo información sobre esto en el documento proporcionado."

Historial:
{historial}

Contexto:
{context}

Pregunta:
{question}

Respuesta:
"""
            prompt = PromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()

            respuesta = chain.invoke({
                "historial": historial_str,
                "context": contexto_str,
                "question": request.pregunta
            })
        else:
            # LLM libre → respuesta directa y concisa
            template = """
Responde de forma clara y breve a la siguiente pregunta.
Historial:
{historial}

Pregunta:
{question}

Respuesta:
"""
            prompt = PromptTemplate.from_template(template)
            chain = prompt | llm | StrOutputParser()

            respuesta = chain.invoke({
                "historial": historial_str,
                "question": request.pregunta
            })

        # Retornar solo texto limpio
        return {"respuesta": respuesta.strip()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en IA: {str(e)}")