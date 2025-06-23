from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

@tool(return_direct=True)
def greet(user_name: str) -> int:
    """Greet user."""
    return f"Hello {user_name}!"

tools = [greet]


primary_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"), 
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
)

fallback_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",  # Use a different model   
    temperature=0.3,
)

llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

agent = create_react_agent(
    model=llm_with_fallback.bind_tools(tools, tool_choice={"type": "tool", "name": "greet"}),
    tools=tools,
)

respnse = agent.invoke({"messages": [{"role": "user", "content": "what's 3 + 5?"}]})
print('**' * 50)
print('--' * 20)
print('User Query')
print(respnse['messages'][0].content)
print('--' * 20)
print('Assistant Response')
print(respnse['messages'][1].content)
print('Response Time', respnse['messages'][1].response_metadata)
print('**' * 50)
