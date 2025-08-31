import os
from mcp.server.fastmcp import FastMCP
import google.auth.transport.requests
import google_auth_oauthlib.flow
import googleapiclient.discovery
from google.oauth2.credentials import Credentials
import logging

app = FastMCP("Google Calendar MCP")
CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar"]
TOKEN_PATH = "token.json"
CREDS_PATH = "credentials.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("calendar_mcp")


@app.tool()
def book_calendar_event(summary: str, start_time: str, end_time: str, timezone: str = "Asia/Kolkata") -> dict:
    """
    Books a new event in Google Calendar.
    """
    creds = None

    # Load existing token
    if os.path.exists(TOKEN_PATH):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_PATH, CALENDAR_SCOPES)
        except Exception:
            logger.warning("Existing token.json invalid, re-running OAuth flow.")
            creds = None

    # If no valid credentials, run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(google.auth.transport.requests.Request())
        else:
            if not os.path.exists(CREDS_PATH):
                raise FileNotFoundError("credentials.json not found. Download from Google Cloud Console.")
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CREDS_PATH, CALENDAR_SCOPES
            )
            creds = flow.run_local_server(port=0)

        # Save new credentials
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())
        logger.info("Saved new token.json")

    service = googleapiclient.discovery.build("calendar", "v3", credentials=creds)

    event = {
        "summary": summary,
        "start": {"dateTime": start_time, "timeZone": timezone},
        "end": {"dateTime": end_time, "timeZone": timezone},
    }

    event_result = service.events().insert(calendarId="primary", body=event).execute()
    return {"event_id": event_result["id"], "htmlLink": event_result.get("htmlLink")}


if __name__ == "__main__":
    logger.info("Starting Calendar MCP server...")
    app.run()
