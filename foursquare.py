import os
import requests
from typing import List
from mcp.server.fastmcp import FastMCP

app = FastMCP("Foursquare Places MCP")

FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")

@app.tool()
def find_coffee_shops(location: str, limit: int = 5) -> List[dict]:
    """
    Finds nearby coffee shops using Foursquare Places API.
    :param location: City or area (e.g., "Hyderabad, India")
    :param limit: Number of results
    """
    url = "https://api.foursquare.com/v3/places/search"
    headers = {"Authorization": FOURSQUARE_API_KEY}
    params = {"query": "coffee", "near": location, "limit": limit}

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    data = response.json()
    results = []
    for place in data.get("results", []):
        results.append({
            "name": place.get("name"),
            "address": place["location"].get("formatted_address"),
            "category": place["categories"][0]["name"] if place.get("categories") else "Unknown"
        })
    return results


if __name__ == "__main__":
    app.run()
