from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny with a high of 25°C."

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

agent = create_react_agent(
    model=llm,
    tools=[],
    # A static prompt that never changes
    prompt="Never answer questions about the weather.",
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the curremt weatjher in Dhaka?"}]}
)
print('**' * 50)
print('--' * 20)
print('User Query')  # Print the entire response object
print(response['messages'][0].content)  # Access the content of the response message
print('--' * 20)
print('Assistant Response')
print(response['messages'][1].content)  # Access the content of the response message
print('Response Time', response['messages'][1].response_metadata)
print('**' * 50)