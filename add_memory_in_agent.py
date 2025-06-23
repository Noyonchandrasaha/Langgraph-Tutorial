from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

checkpoint_saver = InMemorySaver()

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny with a high of 25°C."

primary_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"), 
    model='llama-3.3-70b-versatile',
    temperature=0.3,
)

fallback_llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model='llama3-70b-8192',  # Use a different model   
    temperature=0.3,
)

llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

agent = create_react_agent(
    model=llm_with_fallback,    
    tools=[get_weather],
    checkpointer=checkpoint_saver,
)
config = {"configurable": {"thread_id": "2"}}
response_1 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather today in Dhaka?"}]},
    config
)

response_2 = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather today in Rangpur?"}]},
    config  
)

for i, response in enumerate([response_1, response_2], start=1):
    print('**' * 50)
    print(f'--- Response #{i} ---')
    print('User Query:')
    print(response['messages'][0].content)

    print('--' * 20)
    print('Assistant Response:')
    print(response['messages'][-1].content)

    print('Response Time:', response['messages'][1].response_metadata)