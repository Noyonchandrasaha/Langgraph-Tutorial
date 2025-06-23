from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv  
from pydantic import BaseModel
import os

load_dotenv()

class WeatherResponse(BaseModel):
    conditions: str

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny with a high of 25°C."

primary_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
)


agent = create_react_agent(
    model=primary_llm,
    tools=[get_weather],
    prompt="""You are a helpful assistant. Use the tool 'get_weather' to answer weather-related questions.
When you have the answer, respond and do not call the tool again.""",
    response_format=WeatherResponse  # <-- Use this instead of output_schema
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print('**' * 50)
print('--' * 20)
print('User Query')
print(response['messages'][0].content)
print('--' * 20)
print('Assistant Response')
print(response['messages'][1].content)
print('Response Time', response['messages'][1].response_metadata)
print('Structured Response:', response["structured_response"])
print('**' * 50)
