# geoapify_mcp.py
import os
import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict

app = FastMCP("Geoapify Restaurant MCP")

GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")

@app.tool()
def find_restaurants_nearby(city: str, limit: int = 20) -> List[Dict]:
    """
    Find restaurants in a given city using Geoapify Places API.

    Args:
        city (str): City name to search in (e.g., "Lucknow")
        limit (int): Number of results to fetch (default: 20)

    Returns:
        List[Dict]: List of restaurants with name, address, and place_id
    """
    if not GEOAPIFY_API_KEY:
        return [{"error": "GEOAPIFY_API_KEY not set"}]

    # This is your API call format from the example
    url = (
        "https://api.geoapify.com/v2/places"
        f"?categories=catering.restaurant"
        f"&filter=place:{city.lower()}"
        f"&limit={limit}"
        f"&apiKey={GEOAPIFY_API_KEY}"
    )

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        restaurants = []
        for item in data.get("features", []):
            props = item.get("properties", {})
            restaurants.append({
                "name": props.get("name"),
                "address": props.get("formatted"),
                "place_id": props.get("place_id")
            })
        return restaurants

    except Exception as e:
        return [{"error": str(e)}]


if __name__ == "__main__":
    app.run()
