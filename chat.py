from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("Conectando con la base de datos local y el servidor ia")
embeddings = OllamaEmbeddings(
	model = "nomic-embed-text",
	base_url = "http://192.168.0.10:11434"
)

llm = OllamaLLM(
	model = "phi3",
	base_url = "http://192.168.0.10:11434"

)

vectorstore = Chroma(
	persist_directory = "./db_vectorial",
	embedding_function = embeddings
)

# Solo se toma los 3 fragmentos mas relevantes
retriever = vectorstore.as_retriever(search_kwargs = {"k":3})
template = """Eres un asistente corporativo experto. Usa exclusivamente los siguientes fragmentos de contexto para responder a la pregunta de forma clara y concisa.
Si la respuesta a la pregunta no se encuentra en elcontexto proporcionado, responde exactamente "No tengo informacion sobre esto en el documento proporcionado". No inventes datos

Contexto recuperado:
{context}

pregunta del usuario: {question}

respuesta:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
	 return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
	{"context": retriever | format_docs, "question": RunnablePassthrough()}
	| prompt
	| llm
	| StrOutputParser()
)


print("\n Asistente ROG corporativo")
pregunta_usuario = "Haz un resumen en 3 lineas de los temas principales de este documento"
print(f"\nPregunta:{pregunta_usuario}")

print(f"Generando respuesta... \n")

respuesta = rag_chain.invoke(pregunta_usuario)
print(respuesta)


