import requests, json, os, argparse, sqlite3, time
import pandas as pd
from datetime import datetime, timedelta
from ollama import Client
import mlflow
import requests
from dotenv import load_dotenv
load_dotenv()
import time
from mlflow.exceptions import MlflowException

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
TVH_URL = os.getenv('TVHEADEND_URL')
TVH_AUTH = (os.getenv('TVHEADEND_USER'), os.getenv('TVHEADEND_PASS'))
DB_PATH = os.getenv('DB_PATH', '/data/tv-detection1.db')
CUSTOM_PROMPT = os.getenv('TV_PROMPT', "Is this worth recording for ML training data (news, weather, local events, interesting content)? Answer ONLY Yes or No + short reason.")

if not all([TVH_URL, TVH_AUTH[0], TVH_AUTH[1], OLLAMA_URL, MLFLOW_TRACKING_URI]):
    raise ValueError("Missing required env vars")

ollama_client = Client(host=OLLAMA_URL)
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('CREATE TABLE IF NOT EXISTS schedules (id INTEGER PRIMARY KEY, channel TEXT, start_ts INTEGER, stop_ts INTEGER, reason TEXT, created_at TEXT)')
    conn.commit(); conn.close()

def enrich_with_imdb(title: str) -> str:
    try:
        url = f"http://www.omdbapi.com/?t={requests.utils.quote(title)}&plot=short"
        resp = requests.get(url).json()
        if resp.get('Response') == 'True':
            rating = resp.get('imdbRating', 'N/A')
            genre = resp['Genre']
            year = resp['Year']
            plot = resp['Plot']
            return f"IMDb info: {resp['Title']} ({year}) – Rating: {rating}/10 – Genre: {genre} – Plot: {plot}"
        return "IMDb: No match found"
    except:
        return "IMDb: Lookup failed"

def fetch_epg():
    # Get all channels
    channels_resp = requests.get(f"{TVH_URL}/api/channel/grid?limit=500", auth=TVH_AUTH)
    channels = channels_resp.json().get('entries', [])
    print(f"Found {len(channels)} channels")

    all_events = []
    for ch in channels:
        uuid = ch['uuid']
        name = ch['name']
        # NO start/end – just large limit + asc sort
        url = f"{TVH_URL}/api/epg/events/grid?channel={uuid}&limit=999999&dir=asc"
        resp = requests.get(url, auth=TVH_AUTH).json()
        events = resp.get('entries', [])
        if events:
            print(f"{name}: {len(events)} events")
            all_events.extend(events)
    
    # Filter to future events manually
    now = int(time.time())
    one_hour = now + 3600
    future_events = [e for e in all_events if (e['start'] < one_hour and e['start'] > now)]
    future_events.sort(key=lambda x: x['start'])  # Earliest first
    future_events.sort(key=lambda item: (
                          int(item['channelNumber'].split('.')[0]),  # major number
                          int(item['channelNumber'].split('.')[1])   # subchannel
                      ))
    print(f"Future events (next hour): {len(future_events)}")
    return future_events

def should_record(prompt):
    resp = ollama_client.chat(model='mistral:7b-instruct-q5_K_M', messages=[{'role':'user','content':prompt}])
    reason = resp['message']['content'].strip()
    return 'yes' in reason.lower(), reason

def schedule_recording(event):
    payload = {
        "enabled": True,
        "channelUuid": event['channelUuid'],
        "start": event['start'],
        "stop": event['stop'],
        "title": {"en": event['title']},
        "comment": "tv-detection1 auto"
    }
    r = requests.post(f"{TVH_URL}/api/dvr/entry/create", auth=TVH_AUTH, json=payload)
    return r.status_code == 201

def get_prompt(event, imdb_info):
    prompt = f"""Chicago OTA TV program:
Title: {event['title']}
Description: {event.get('description','')}
Summary: {imdb_info}

{CUSTOM_PROMPT}
"""
    return prompt

def log_mlflow(events_checked, scheduled, results):
    try:
        mlflow.set_experiment("tv-detection1")
        with mlflow.start_run():
            mlflow.log_param("mode", "schedule")
            mlflow.log_param("custom_prompt", CUSTOM_PROMPT)
            mlflow.log_param("events_checked", events_checked)
            mlflow.log_param("scheduled", scheduled)
            mlflow.log_metric("success_rate", scheduled / max(1, events_checked))
            if results:
                df = pd.DataFrame(results)
                
                # Optional: sort by start time for nicer view
                df['start'] = pd.to_datetime(df['start'])
                df = df.sort_values('start').reset_index(drop=True)
                
                # Log as interactive table
                mlflow.log_table(df, artifact_file="decisions")
                
                # Optional: also log as CSV for download
                mlflow.log_table(df, artifact_file="decisions.csv")


    except MlflowException as e:
        if "NameResolutionError" in str(e):
            print(f"MLFlow DNS failure: {e} – skipping logging (run incomplete)")
        else:
            print(f"MLFlow error: {e} – skipping")
    except Exception as e:
        print(f"Unexpected MLFlow log error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='schedule', choices=['schedule', 'test-epg'])
    args = parser.parse_args()
    init_db()

    if args.mode == 'schedule':
        print(f"=+=+=+=+=+=+=+=+ Filtering EPG events starting in the next hour based on this prompt: {CUSTOM_PROMPT} =+=+=+=+=+=+=+=+")
        summary_obj = {'prompt': CUSTOM_PROMPT, 'shows': []}
        events = fetch_epg()  # future_events list
        events_checked = len(events)
        scheduled = 0
        all_show_objects = []
        for event in events:
            start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event['start']))
            imdb_info = enrich_with_imdb(event['title'])
            prompt = get_prompt(event, imdb_info)
            yes, reason = should_record(prompt)
            show_obj = {
                'channel': event.get('channelNumber', 'N/A'),
                'start': start_time,
                'title': event['title'],
                'reason': reason
            }
            if yes:
                if schedule_recording(event):
                    # DB insert if you have it
                    scheduled += 1
                    show_obj['action'] = 'scheduled ok'
                else:
                    show_obj['action'] = 'schedule failed'
                print(f"====Scheduled: Channel {event['channelNumber']}   {start_time}   {event['title']} - {reason}")
            else:
                show_obj['action'] = 'skipped'
                print(f"====Skipped: Channel {event['channelNumber']}   {start_time}   {event['title']} - {reason}")
            all_show_objects.append(show_obj)
        log_mlflow(events_checked, scheduled, all_show_objects)
        print(f"Checked {events_checked} events, scheduled {scheduled}")
    elif args.mode == 'test-epg':
      events = fetch_epg()
      if events:
          print(json.dumps(events[:3], indent=2))
      else:
          print("No future events found")
