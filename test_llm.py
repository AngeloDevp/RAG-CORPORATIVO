from langchain_community.llms import Ollama

print("Conectando a la asus rog zephyrus g14...")

llm = Ollama(model="phi3", base_url="http://192.168.0.10:11434")

respuesta = llm.invoke("Hola eres el cerebro de mi nuevo sistema corporativo. Responde en un parrafo corto ¿estas listo para trabajar?")

print("respuesta de phi3")

print(respuesta)
