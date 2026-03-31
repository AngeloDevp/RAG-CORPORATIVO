from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("cargando el documento pdf")
loader = PyPDFLoader("documentos/prueba.pdf")
documento = loader.load()

print("dividiendo el texto en fragmentos pequegnos")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
fragmentos = text_splitter.split_documents(documento)
print(f"el documento se dividio en {len(fragmentos)} fragmentos")

print("Conectando con la asus para generar el embeddings")
embeddings = OllamaEmbeddings(
	model = "nomic-embed-text",
	base_url = "http://192.168.0.10:11434"
)

print("Guardando en la base de datos vectorial")
vectorstore = Chroma.from_documents(
	documents = fragmentos,
	embedding = embeddings,
	persist_directory = "./db_vectorial"
)

print("El documento ha sido vectorizado y guardado de forma segura")

