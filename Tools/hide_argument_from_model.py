from typing import Annotated
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def do_something_with_state(messages):
    # Example: just print or log state messages
    print("State messages:", messages)

def do_something_with_config(config):
    # Example: just print or log config info
    print("Config info:", config)

@tool
def my_tool(
    tool_arg: str,
    state: Annotated[AgentState, InjectedState],
    config: RunnableConfig,
) -> str:
    """My tool that uses injected state and config, but model only sees tool_arg."""
    do_something_with_state(state["messages"])
    do_something_with_config(config)
    return f"Processed tool_arg: {tool_arg}"

primary_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
)

fallback_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
    temperature=0.3,
)

llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

agent = create_react_agent(
    model=llm_with_fallback,
    tools=[my_tool],
)

response = agent.invoke({"messages": [{"role": "user", "content": "test tool usage"}]})
print("Response messages:", response['messages'])

print("Assistant response:", response['messages'][-1].content)
