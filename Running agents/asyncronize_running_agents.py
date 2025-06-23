from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny with a high of 25°C."

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

async def main():
    agent = create_react_agent(
        model=llm_with_fallback,
        tools=[get_weather],
    )

    response = await agent.ainvoke({"messages": [{"role": "user", "content": "what is the weather in sf"}]})

    print('**' * 50)
    print('--' * 20)
    print('User Query')
    print(response['messages'][0].content)
    print('--' * 20)
    print('Assistant Response')
    print(response['messages'][2].content)
    print('Response Time', response['messages'][1].response_metadata)
    print('**' * 50)

if __name__ == "__main__":
    asyncio.run(main())