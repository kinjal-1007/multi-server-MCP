# Book My Calendar (MCP Multi-Server Demo)

This project shows how to use a **Model Context Protocol (MCP) client** with multiple servers (Calendar, Krutrim Places, and Geoapify) and a **Groq LLM**.  
The assistant can dynamically invoke multiple tools in a single conversation — for example, finding restaurants and then booking a meeting in Google Calendar.

---

## Features

- **Multiple MCP servers**: Calendar, Krutrim Places, Geoapify  
- **Dynamic tool discovery**  
- **Chained tool invocation**: find → confirm → book  
- **Google Calendar integration** via OAuth2  
- **Groq LLM** for reasoning and tool selection  

---

## Setup

1. **Navigate to the project directory:**
```bash
cd /path/to/this/folder
```

2. **Initialize the project and create a virtual environment:**
```bash
uv init kafka-mcp
uv venv
```

3.  **Install the dependencies from the provided `requirements.txt`:**
```bash
uv add -r requirements.txt
```

4. **Environment Variables:**

Create a `.env` file in the root folder:

```
GROQ_API_KEY=your_groq_api_key
GEOAPIFY_API_KEY=your_geoapify_key
KRUTRIM_API_KEY=your_krutrim_key
```

5. **Google Calendar Setup:**

1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project and enable the **Google Calendar API**.
3. Go to **APIs & Services > Credentials**.
4. Create an **OAuth Client ID** (Desktop app).
5. Download the `credentials.json` file and place it in the project root.
6. On first run, you will be prompted to log in with your Google account.

   * A `token.json` file will be created for future use.

---

## Running the Client

This demo uses three MCP servers and one client.
Each server must be started before running the client.

Start Servers (in separate terminals):
```bash
python calendar_server.py
python krutrim_server.py
python geoapify_server.py
```

Start the MCP client:

```bash
python client.py
```

---

## Example Conversation

Below is a real flow showing **multiple tools invoked in sequence**:

```
You: I want to have lunch at 2 PM tomorrow with friends. Find suitable restaurants. My location is Lucknow (26.846259, 80.949000).

Assistant: Here are the top restaurants near your location in Lucknow:
1. Shyama Chauraha (111m)
2. Rovers (260m)
3. Cafe Coffee Day (300m)
4. Curry Leaf Hazratganj (352m)
5. La Pino'z Pizza (854m)

Would you like me to book a calendar event for lunch tomorrow at 2:00 PM (Asia/Kolkata) at one of these restaurants?

You: I pick La Pino'z

Assistant: Your lunch meeting at La Pino'z Pizza has been scheduled for Tomorrow (Aug 23, 2025) at 2:00 PM (1 hour).

Calendar Event Link: https://www.google.com/calendar/event?eid=OWN2Nm1tcW11YWdvdDQ1NDM3bm1ocmt1YWsga2luamFsLnRhYmxldGE5QG0
```

This demonstrates how the assistant:

1. Uses **Geoapify/Krutrim** to fetch nearby restaurants.
2. Confirms the user’s choice.
3. Calls the **Calendar MCP server** to book the event.
4. Returns a Google Calendar link for the scheduled meeting.

---

## Troubleshooting

* **Invalid Credentials**: Delete `token.json` and restart to re-authenticate.
* **Missing API Keys**: Ensure `.env` contains Groq, Geoapify, and Krutrim keys.
* **Datetime Issues**: Use ISO 8601 format (`YYYY-MM-DDTHH:MM:SS+TZ`).

---

## License
