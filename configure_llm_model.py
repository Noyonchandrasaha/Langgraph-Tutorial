from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny with a high of 25°C."
#add a temperature parameter to the model

primary_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_tokens=1024,
    disable_streaming=True
)
fallback_llm = ChatGroq(
    model="llama3-70b-8192", # Use a different model for fallback
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3,
    max_tokens=524
)
llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

agent = create_react_agent(
    model=llm_with_fallback,
    tools=[get_weather],
    prompt="You are a helpful assistant that can provide weather information using the get_weather function."
)

response = agent.invoke({"messages": [{"role": "user", "content": "What is the weather today in Dhaka?"}]})
print('**' * 50)
print('--' * 20)
print('User Query')  # Print the entire response object
print(response['messages'][0].content)  # Access the content of the response message
print('--' * 20)
print('Assistant Response')
print(response['messages'][2].content)  # Access the content of the response message
print('Response Time', response['messages'][1].response_metadata)
print('**' * 50)