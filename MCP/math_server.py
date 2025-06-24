# math_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math")

@mcp.tool()
async def add_numbers(a: float, b: float) -> str:
    return f"The sum of {a} and {b} is {a + b}."

@mcp.tool()
async def multiply_numbers(a: float, b: float) -> str:
    return f"The product of {a} and {b} is {a * b}."

if __name__ == "__main__":
    mcp.run(transport="stdio")
