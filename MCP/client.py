import asyncio
import os
import sys
from typing import Optional
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

load_dotenv()

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        # ✅ Replace with your Groq model and key
        self.chat_model = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY")
        )

    async def connect_to_server(self, server_script_path: str):
        command = "python"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()
        response = await self.session.list_tools()
        tools = response.tools
        print("\n✅ Connected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        messages = [
            SystemMessage(content="You are a helpful assistant that can call tools like add_numbers or greet."),
            HumanMessage(content=query)
        ]

        response = await self.session.list_tools()
        available_tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in response.tools
        ]

        while True:
            ai_response = await self.chat_model.ainvoke(messages, tools=available_tools)

            if isinstance(ai_response, AIMessage) and ai_response.tool_calls:
                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    print(f"[🛠] Calling tool: {tool_name} with args: {tool_args}")
                    result = await self.session.call_tool(tool_name, tool_args)
                    print(f"[✅] Tool result: {result.content}")

                    # 🔥 Fix: Ensure result is a string
                    if isinstance(result.content, list):
                        result_text = "\n".join([c.text for c in result.content if hasattr(c, "text")])
                    else:
                        result_text = str(result.content)

                    messages.append(ai_response)
                    messages.append(ToolMessage(tool_call_id=tool_id, content=result_text))

            elif isinstance(ai_response, AIMessage):
                return ai_response.content or "[🤖 No response from assistant.]"
            else:
                return "[🤖 Unexpected response type.]"

    async def chat_loop(self):
        print("\n🤖 MCP Client Started! Type your query or 'quit' to exit.")
        while True:
            try:
                query = input("\nYou: ").strip()
                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\nAssistant:\n", response)
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
