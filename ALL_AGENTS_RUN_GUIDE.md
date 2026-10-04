# Smart Water Management: all agents

## What runs

| Service | Port | Role |
| --- | ---: | --- |
| Router and Waterwise dashboard | 8000 | Receives UI requests and coordinates agents |
| Collector | 8001 | Supplies simulated live readings and historical data |
| Analysis | 8002 | Detects usage anomalies and forecasts demand |
| Guidance (IR) | 8003 | Searches policy and procedure documents |
| Alert | 8004 | Flags anomalies and shortage risk from analysis results |
| Summarizer | 8005 | Turns analysis results into readable summaries |

## Data flow

```text
Browser → Router → Collector → Analysis → Summarizer
                                   └──────→ Alert
                         └────────→ Guidance when an anomaly needs evidence

Browser → Router → Analysis → Summarizer + Alert     (forecast)
Browser → Router → Guidance                          (document search)
Browser → Router → complaint classification + Guidance (complaint)
```

The Alert Agent creates an alert result in the dashboard. It does not send email, SMS, or push notifications. The Summarizer uses Ollama with `llama3.2:3b` when available and returns its built-in rule based summary when Ollama is unavailable.

## Run on Windows

From this project folder, create a Python environment and install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-all.txt
```

The shared configuration is in `router-agent\.env`. The router sets a local browser session when you open the dashboard, so the UI uses the same key without embedding it in its source. Keep your own `.env` private. The prepared friend ZIP includes a working local sample `.env` instead of your private settings.

After installing dependencies, you can start all six services with one command:

```powershell
.\start_all.ps1
```

If PowerShell blocks scripts on your laptop, use the individual commands below.

Start each service in its own PowerShell window from this project folder:

```powershell
.\venv\Scripts\python.exe -m uvicorn main:app --app-dir collector-agent --host 127.0.0.1 --port 8001
.\venv\Scripts\python.exe -m uvicorn main:app --app-dir iwra\analysis-agent --host 127.0.0.1 --port 8002
.\venv\Scripts\python.exe -m uvicorn main:app --app-dir ir-module --host 127.0.0.1 --port 8003
.\venv\Scripts\python.exe -m uvicorn main:app --app-dir alert-agent --host 127.0.0.1 --port 8004
.\venv\Scripts\python.exe -m uvicorn main:app --app-dir summarizer-agent --host 127.0.0.1 --port 8005
.\venv\Scripts\python.exe -m uvicorn main:app --app-dir router-agent --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000/**. All six status indicators should show Online. The dashboard itself is served by the Router, so no separate web server is needed.

## Try the complete flow

1. Select **West** and **7 days**, then click **Generate forecast**. The chart, summary, and shortage alert result come from Analysis, Summarizer, and Alert.
2. Click **Analyze reading**. Collector supplies the reading; Analysis scores it; Summarizer explains it; Alert checks whether it needs attention.
3. Search **leak procedure** in Guidance library to see retrieved documents.
4. Send **There is a pipe leak in Zone 4** in the assistant to see complaint priority and guidance.

To use Ollama summaries, install Ollama and pull `llama3.2:3b`. This is optional for the six services to work.

## Share with a friend

Run `python make_friend_zip.py` from this project folder and send `Waterwise_All_Agents.zip`. The ZIP includes all six agents, the model and data, the dashboard, a local sample `.env`, and the run guide. It excludes installed virtual environments, caches, logs, and this computer's private `.env`.
