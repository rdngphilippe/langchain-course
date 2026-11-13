from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from tavily import TavilyClient

load_dotenv()
tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """

    print(f"Searching for query: {query}")
    return tavily.search(query=query)

    
llm = ChatOllama(model="llama3.2:3b")
tools = [search]
agents = create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from langchain-course!")
    result = agents.invoke({"messages": HumanMessage(content="What is the weather in Tokyo?")})
    print(result)

if __name__ == "__main__":
    main()