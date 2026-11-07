from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2:3b", temperature=0.7)

response = llm.invoke("Hello, how are you?")
print(response.content)
