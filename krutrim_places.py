# krutrim_mcp.py
import os
import requests
from mcp.server.fastmcp import FastMCP
from typing import List, Dict

app = FastMCP("Ola Krutrim Restaurant MCP")

OLAKRUTRIM_API_KEY = os.getenv("OLAKRUTRIM_API_KEY")

@app.tool()
def find_restaurants_nearby(lat: float, lon: float, radius: int = 1000) -> List[Dict]:
    """
    Find restaurants near a given lat/lon using Ola Krutrim Places API.

    Args:
        lat (float): Latitude
        lon (float): Longitude
        radius (int): Search radius in meters (default: 1000)

    Returns:
        List[Dict]: List of restaurants with name, address, place_id, and distance
    """
    if not OLAKRUTRIM_API_KEY:
        return [{"error": "OLAKRUTRIM_API_KEY not set"}]

    url = (
        "https://api.olamaps.io/places/v1/nearbysearch"
        f"?location={lat},{lon}"
        "&types=restaurant"
        f"&radius={radius}"
        f"&api_key={OLAKRUTRIM_API_KEY}"
    )

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        restaurants = []
        for item in data.get("predictions", []):
            structured = item.get("structured_formatting", {})
            restaurants.append({
                "name": structured.get("main_text"),
                "address": structured.get("secondary_text"),
                "place_id": item.get("place_id"),
                "distance_meters": item.get("distance_meters")
            })
        return restaurants

    except Exception as e:
        return [{"error": str(e)}]


if __name__ == "__main__":
    app.run()
