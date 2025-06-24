from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # This is a placeholder implementation. Replace with actual weather API call.
    return f"The weather in {location} is sunny with a high of 25°C."