from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
# from serpapi import GoogleSearch

load_dotenv()

@tool
def web_search(query: str) -> str:
    """Run a simple web search and return a summary result."""
    params = {"q": query, "api_key": os.getenv("SERPAPI_KEY")}
    # results = GoogleSearch(params).get_dict().get("organic_results", [])
    # return results[0]["snippet"] if results else "No results found."

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0.3
)

agent = create_react_agent(
    model=llm,
    tools=[web_search],
)

response = agent.invoke(
    {"messages": [{"role":"user", "content": "What was a positive news story from today?"}]}
)

print(response['messages'][-1].content)
