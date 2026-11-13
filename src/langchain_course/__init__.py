from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from tavily import TavilyClient
from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The answer from the agent to the query")
    sources: List[Source] = Field(default_factory=list, description="The list of sources used to generate the answer")

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

    
llm = ChatOllama(model="qwen3:8b")
tools = [TavilySearch()]
# Ollama ignores tool_choice, so the model must be told explicitly to finish
# by calling the AgentResponse tool, otherwise structured_response stays None.
SYSTEM_PROMPT = (
    "You are a helpful research assistant. Use the tavily_search tool to find information. "
    "When you have the information, you MUST give your final answer by calling the "
    "AgentResponse tool with the full answer and the list of source URLs. "
    "Never reply with plain text."
)

agents = create_agent(
    model=llm,
    tools=tools,
    response_format=ToolStrategy(AgentResponse),
    system_prompt=SYSTEM_PROMPT,
)

def main() -> None:
    print("Hello from langchain-course!")
    result = agents.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Brussels in Belgium on linkedin and list their details?")]})
    print(result["structured_response"])

if __name__ == "__main__":
    main()