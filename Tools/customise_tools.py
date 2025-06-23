from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

class multipleInputSchema(BaseModel):
    a: int = Field(..., description="The first number to add.")
    b: int = Field(..., description="The second number to add.")

@tool("multiply", args_schema=multipleInputSchema)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


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
    model=llm_with_fallback,
    tools=[multiply],
)

respnse = agent.invoke({"messages": [{"role": "user", "content": "what is 5000 and 4000"}]})
print('**' * 50)
print('--' * 20)
print('User Query')
print(respnse['messages'][0].content)
print('--' * 20)
print('Assistant Response')
print(respnse['messages'][2].content)
print('Response Time', respnse['messages'][1].response_metadata)
print('**' * 50)
