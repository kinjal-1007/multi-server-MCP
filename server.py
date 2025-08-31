import os
import requests
import datetime
from typing import List
from mcp.server.fastmcp import FastMCP

# Google Calendar imports
import google.auth.transport.requests
import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from google.oauth2.credentials import Credentials

app = FastMCP("Coffee+Calendar MCP")

FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]

# ----------------- Foursquare Tool -----------------
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


# ----------------- Google Calendar Tool -----------------
@app.tool()
def book_calendar_event(summary: str, start_time: str, end_time: str, timezone: str = "Asia/Kolkata") -> dict:
    """
    Books a new event in Google Calendar.
    :param summary: Event name (e.g., "Coffee meeting")
    :param start_time: Start time in ISO format (e.g., "2025-08-21T10:00:00")
    :param end_time: End time in ISO format (e.g., "2025-08-21T11:00:00")
    :param timezone: Timezone (default: Asia/Kolkata)
    """
    creds = None
    token_path = "token.json"

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, CALENDAR_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                "credentials.json", CALENDAR_SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    service = googleapiclient.discovery.build("calendar", "v3", credentials=creds)

    event = {
        "summary": summary,
        "start": {"dateTime": start_time, "timeZone": timezone},
        "end": {"dateTime": end_time, "timeZone": timezone},
    }

    event_result = service.events().insert(calendarId="primary", body=event).execute()
    return {"event_id": event_result["id"], "htmlLink": event_result.get("htmlLink")}


if __name__ == "__main__":
    app.run()

