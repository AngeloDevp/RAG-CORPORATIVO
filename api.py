import os
from pydantic import BaseModel
from typing import List, Dict, Any
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
# Importa aquí las librerías de tu RAG (ChromaDB, LangChain, etc.)

# 1. Variable global para mantener UNA SOLA conexión a la base de datos
vector_db_client = None

def cargar_base_de_datos(ruta_db: str):
    """
    Función que inicializa tu base de datos vectorial.
    Sustituye el contenido de esta función con tu código real.
    """
    print(f"📂 Detectando/Cargando base de datos en: {ruta_db}")
    # EJEMPLO CON CHROMADB:
    # import chromadb
    # client = chromadb.PersistentClient(path=ruta_db)
    # collection = client.get_or_create_collection(name="rag_corporativo")
    # return collection
    
    # Mock (Simulación) para este ejemplo
    class MockDB:
        def procesar(self, archivo):
            print(f"⚙️ Vectorizando e insertando {archivo} en la base de datos...")
    return MockDB()


# 2. Configuración del "Lifespan" (El patrón Singleton de FastAPI)
@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db_client
    
    # RUTA SEGURA: Recomiendo usar rutas absolutas. 
    # Si 'Documents' tiene OneDrive, considera cambiar esto a 'C:/Proyectos_Locales/db_api'
    db_path = os.path.abspath("./db_api")
    os.makedirs(db_path, exist_ok=True)
    
    # --- STARTUP (Arranque del servidor) ---
    try:
        vector_db_client = cargar_base_de_datos(db_path)
        print("✅ Base de datos cargada correctamente.")
        yield
    finally:
        # --- SHUTDOWN (Apagado del servidor) ---
        print("🛑 Cerrando conexiones de manera segura...")
        # Si tu cliente de DB tiene un método close(), llámalo aquí.
        # Ejemplo: vector_db_client.close()
        vector_db_client = None


# 3. Inicialización de la App usando el lifespan
app = FastAPI(lifespan=lifespan)


# 4. Endpoint de subida de archivos blindado contra WinError 32
@app.post("/upload")
async def procesar_archivo(file: UploadFile = File(...)):
    if not vector_db_client:
        raise HTTPException(status_code=500, detail="La base de datos no está disponible.")

    temp_file_path = f"./temp_{file.filename}"
    
    try:
        # Guardar el archivo subido a disco de forma segura usando un bloque 'with'
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"📄 Archivo temporal guardado en: {temp_file_path}")

        # Aquí llamas a la inserción de tu RAG usando el cliente GLOBAL
        vector_db_client.procesar(temp_file_path)
        
        return {"mensaje": f"Archivo '{file.filename}' vectorizado con éxito."}

    except Exception as e:
        # Si algo falla (ej. error 500), capturamos el error para que el bloque 'finally' se ejecute
        print(f"❌ Error durante el procesamiento: {e}")
        raise HTTPException(status_code=500, detail=f"Error procesando archivo: {str(e)}")

    finally:
        # --- ESTO ES LO QUE EVITA EL WINERROR 32 ---
        
        # 1. Liberamos el archivo subido de la memoria de FastAPI
        await file.close()
        
        # 2. Eliminamos el archivo temporal que creamos para no dejar "basura" bloqueada
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print("🧹 Archivo temporal limpiado.")
            except PermissionError:
                # Si esto ocurre, una librería síncrona (como PyPDFLoader) no cerró el archivo internamente.
                print(f"⚠️ Advertencia: No se pudo eliminar {temp_file_path}. Otra librería lo sigue usando.")

# 1. Creamos un "Modelo" que represente el JSON que envía Streamlit
class PeticionChat(BaseModel):
    pregunta: str
    historial: List[Dict[str, Any]] = [] # Permite recibir el historial de mensajes vacío o lleno

# 5. Endpoint de Chat reutilizando la misma conexión
@app.post("/chat")
async def chat_rag(peticion: PeticionChat):
    if not vector_db_client:
        raise HTTPException(status_code=500, detail="La base de datos no está disponible.")
    
    # Extraemos la información del objeto que nos llegó
    pregunta_usuario = peticion.pregunta
    historial_chat = peticion.historial
    
    # Imprimimos en la consola de la ASUS para confirmar que llegó bien
    print(f"Recibido - Pregunta: {pregunta_usuario} | Historial: {len(historial_chat)} mensajes previos")
    
    # Aquí iría tu lógica de LangChain usando la variable "pregunta_usuario"
    # respuesta = vector_db_client.query(query_texts=[pregunta_usuario])
    
    # Por ahora, retornamos un JSON normal
    return {"respuesta": f"Respuesta simulada para: {pregunta_usuario}"}