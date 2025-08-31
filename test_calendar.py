from datetime import datetime, timedelta
import requests

# Sample test parameters
summary = "Test Meeting"
start_time = (datetime.now() + timedelta(minutes=5)).isoformat()
end_time = (datetime.now() + timedelta(minutes=65)).isoformat()

# Use requests if the MCP server exposes HTTP (if using FastMCP stdio, use an MCP client)
# For direct Python test:
from calendar_server import book_calendar_event

result = book_calendar_event(summary, start_time, end_time)
print(result)
