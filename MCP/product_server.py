import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("product")

FAKESTORE_API_URL = "https://fakestoreapi.com/products"

@mcp.tool()
async def list_products() -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(FAKESTORE_API_URL, timeout=20)
        resp.raise_for_status()
        products = resp.json()
    titles = [f"{p['id']}: {p['title']}" for p in products]
    return "Available products:\n" + "\n".join(titles)

@mcp.tool()
async def get_product_by_id(id: int) -> str:
    url = f"{FAKESTORE_API_URL}/{id}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=20)
        if resp.status_code == 404:
            return "Product not found."
        resp.raise_for_status()
        product = resp.json()

    return (
        f"Title: {product['title']}\n"
        f"Price: ${product['price']}\n"
        f"Description: {product['description']}\n"
        f"Category: {product['category']}\n"
        f"Rating: {product['rating']['rate']} ({product['rating']['count']} reviews)\n"
        f"Image URL: {product['image']}"
    )

if __name__ == "__main__":
    mcp.run(transport="stdio")
