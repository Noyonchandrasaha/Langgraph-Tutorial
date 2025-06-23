from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny with a high of 25°C."
#add a temperature parameter to the model
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
    temperature=0.3,  # Adjust the temperature for more controlled responses,
    max_tokens=1024  # Set a maximum token limit for the response
)

agent = create_react_agent(
    model=llm,
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