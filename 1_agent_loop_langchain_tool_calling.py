from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


MAX_ITERATIONS = 10
MODEL = "bonsai-2:27b"

@tool
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f" >> Executing get_product_price('{product_name}')")
    prices = {"computer": 42.0, "laptop": 1000.0, "phone": 500.0}
    return prices.get(product_name, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to the given price based on the discount tier.
    Available tiers are 'bronze', 'silver', and 'gold'."""
    discount_percentage = {"bronze": 5, "silver": 10, "gold": 23}
    print(f" >> Executing apply_discount({price}, '{discount_tier}')")
    discount = discount_percentage.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

def run_agent(question:str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question : {question}")
    print("=" * 60)
    messages = [SystemMessage(content=("You are a helpful shop assistant."
                                       "You have access to a product catalog tool "
                                       "and a discount tool. \n\n"
                                       "STRICT RULES - you must follow thse exactly:\n"
                                       "1. NEVER guess or assume any product price."
                                       "You MUST call get_product_price first to get the real price.\n"
                                       "2. Only call apply_discount AFTER you have received "
                                       "a price from get_product_price. Pass the exact price "
                                       "returned by get_product_price - do NOT pass a made-up number.\n"
                                       "3. NEVER calculate discounts yourself using math. "
                                       "Alwyays use the apply_discount tool.\n"
                                       "4. If the user does not specify a discount tier, "
                                       "ask them which tier to use - do NOT assume one.")),                        
                 HumanMessage(content=question)]
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n---Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content
        
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"  [Tool selected] : {tool_name} with args {tool_args} (id: {tool_call_id})")
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found.")
        observation = tool_to_use.invoke(tool_args)
        print(f" [Tool Result]: {observation}")
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    print("ERROR: Maximum iterations reached without a final answer.")
    return None
    


if __name__ == "__main__":
    print(f"Hello LangChain Agent (.bind_tools) !")
    print()
    result = run_agent("What is the price of a computer with a silver discount?")
    print(f"Result: {result}")
