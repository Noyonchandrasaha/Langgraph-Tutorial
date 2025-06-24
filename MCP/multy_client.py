from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["math_server.py"],
                "transport": "stdio",
            },
            "product": {
                "command": "python",
                "args": ["product_server.py"],
                "transport": "stdio"
            }
        }
    )

    tools = await client.get_tools()  # <--- await here

    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192"
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="Do not provide any information that is not related to the tools. You can only use the tools provided to answer the user's queries.",
    )

    user_queries = [
        "Add 5 and 7",
        "Find the product under 500?",
        "Multiply 8 and 12",
        "What is the capital of Bangladesh"
    ]

    for query in user_queries:
        response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})

        # print("DEBUG - Response Type:", type(response))
        # print("DEBUG - Full Response:", response)

        # Get the final assistant message
        final_message = response["messages"][-1]
        print(f"\nUser: {query}\nAssistant: {final_message.content}\n")
        # response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
        # print(f"User: {query}\nAssistant: {response.content}\n")
        # response = await agent.ainvoke({"messages": [{"role": "user", "content": query}]})
        # final_content = response[-1].content
        # print("Assistant:", final_content)
        # print(f"User: {query}\nAssistant: {response['messages']}\n")

if __name__ == "__main__":
    asyncio.run(main())
