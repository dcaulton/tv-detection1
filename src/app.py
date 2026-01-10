import requests, json, os, argparse, sqlite3, time
import pandas as pd
from datetime import datetime, timedelta
from ollama import Client
import mlflow
import requests
from dotenv import load_dotenv
load_dotenv()
import time
import json
from mlflow.exceptions import MlflowException

DAYS_TO_FETCH=4
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
TVH_URL = os.getenv('TVHEADEND_URL')
TVH_AUTH = (os.getenv('TVHEADEND_USER'), os.getenv('TVHEADEND_PASS'))
DB_PATH = os.getenv('DB_PATH', '/data/tv-detection1.db')
CUSTOM_PROMPT = os.getenv('TV_PROMPT', "Is this worth recording for ML training data (news, weather, local events, interesting content)? Answer ONLY Yes or No + short reason.")
SD_USER = os.getenv('SD_USER')
SD_PASS = os.getenv('SD_PASS')
SD_URL = "https://json.schedulesdirect.org/20141201"

if not all([TVH_URL, TVH_AUTH[0], TVH_AUTH[1], OLLAMA_URL, MLFLOW_TRACKING_URI]):
    raise ValueError("Missing required env vars")

ollama_client = Client(host=OLLAMA_URL)
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def init_db():
    print('initializing db')

    create_channel = """
CREATE TABLE IF NOT EXISTS channel (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id TEXT NOT NULL,          -- e.g. "I30415.json.schedulesdirect.org"
    channel TEXT NOT NULL,             -- e.g. "2.1"
    UNIQUE(station_id, channel)
);
"""
    create_schedule = """
CREATE TABLE IF NOT EXISTS schedule (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    program_id INTEGER NOT NULL,
    start_date TEXT NOT NULL,          -- ISO 8601: '2026-01-12T06:00:00+0000'
    end_date TEXT NOT NULL,            -- ISO 8601
    FOREIGN KEY (channel_id) REFERENCES channel(id) ON DELETE CASCADE,
    FOREIGN KEY (program_id) REFERENCES program(id) ON DELETE CASCADE,
    UNIQUE(channel_id, start_date)
);
"""
    create_program = """
CREATE TABLE IF NOT EXISTS program (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sd_programid TEXT NOT NULL,        -- e.g. "EP017254570124"
    title TEXT NOT NULL,
    description TEXT,
    genres TEXT,                       -- comma-separated, e.g. "series,Drama,Crime"
    original_air_date TEXT,            -- ISO 8601 date only: '2025-01-10'
    season INTEGER,                    -- nullable
    episode INTEGER,                   -- nullable
    UNIQUE(sd_programid)
);
"""
    create_person = """
CREATE TABLE IF NOT EXISTS person (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_id INTEGER NOT NULL,
    role TEXT NOT NULL,                -- e.g. "actor", "director", "writer"
    name TEXT NOT NULL,
    character_name TEXT,               -- nullable, only for actors
    sd_name_id TEXT,                   -- nullable, Schedules Direct person ID if available
    FOREIGN KEY (program_id) REFERENCES program(id) ON DELETE CASCADE
);
"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(create_channel)
    conn.execute(create_schedule)
    conn.execute(create_program)
    conn.execute(create_person)
    conn.commit(); 
    conn.close()
    print('db init complete')

def add_channels(channels):
    add_counter = 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("select station_id from channel;")
    rows = cursor.fetchall()
    cur_chan_station_ids = [x[0] for x in rows]
    cursor = conn.cursor()
    for channel in channels:
      if channel.get('stationID') not in cur_chan_station_ids:
        sid = channel.get('stationID')
        channel_dot_number = channel.get('channel')
        cursor.execute(f"INSERT INTO channel (station_id, channel) VALUES (?, ?);", (sid, channel_dot_number))
        add_counter += 1
#        print(f'adding channel {channel_dot_number}')
    conn.commit(); 
    conn.close()
    print(f'{len(channels)} channels processed, {add_counter} added')

def get_channels(conn):
    cursor = conn.execute("select id, station_id from channel;")
    rows = cursor.fetchall()
    channels = {}
    for x in rows:
      channels[x[0]] = x[1]  # key = sql id of the channel,    value = station id of the channel in schedule direct
#    print(f"current channels are {channels}")
    return channels

def get_programs(conn):
    cursor = conn.execute("select id, sd_programid from program;")
    rows = cursor.fetchall()
    programs = {}
    for x in rows:
      programs[x[0]] = x[1]  # key = sql id of the program,    value = program id of the channel in schedule direct
    print(f"current programs are {programs}")
    return programs

def get_schedule_dates(conn):
    cursor = conn.execute("select id, start_date from schedule;")
    rows = cursor.fetchall()
    cur_schedules = {}
    for x in rows:
      sched_id = x[0]
      start_date = x[1]
      if sched_id not in cur_schedules:
        cur_schedules[sched_id] = []
      if start_date not in cur_schedules[sched_id]:
        cur_schedules[sched_id].append(start_date)
    print(f"current schedules are {cur_schedules}")
    return cur_schedules

def get_schedule_dates_by_chan_id(conn):
    cursor = conn.execute("select channel_id, start_date from schedule;")
    rows = cursor.fetchall()
    cur_schedules = {}
    for x in rows:
      chan_id = x[0]
      start_date = x[1]
      if chan_id not in cur_schedules:
        cur_schedules[chan_id] = []
      if start_date not in cur_schedules[chan_id]:
        cur_schedules[chan_id].append(start_date)
    print(f"current schedules by chan id are {cur_schedules}")
    return cur_schedules

def get_program_sd_programids(conn):
    cursor = conn.execute("select sd_programid from program;")
    rows = cursor.fetchall()
    program_ids = []
    for x in rows:
      program_ids.append(x[0])
#    print(f"current program sd ids are {program_ids}")
    return program_ids

def add_schedules(schedules):
    conn = sqlite3.connect(DB_PATH)
    cur_channels = get_channels(conn)
    schedule_dates_by_chan_id = get_schedule_dates_by_chan_id(conn)

    cursor = conn.cursor()
    for schedule in schedules:
      sched_station_id = schedule.get('stationID')
      print(f'new schedule record for station id: {sched_station_id}')
      if sched_station_id not in cur_channels.values():
        print(f'skipping schedule {sched_station_id} - it is for a channel we dont know about')
        continue
      chan_id = [key for key,val in cur_channels.items() if val == sched_station_id][0] # doing this in memory to avoid a flood of sql calls

      for program in schedule.get('programs', []):
        sched_start_date = program.get('airDateTime')
        if chan_id in schedule_dates_by_chan_id and sched_start_date in schedule_dates_by_chan_id[chan_id]:
          print(f'  skipping preexisting schedule for {sched_station_id}  {sched_start_date}')
          continue  # already scheduled
        # TODO build the sched end date and test on date in range, also add end date to table below
#        print(f'adding to schedule ===== {sched_station_id}  {sched_start_date}')
        cursor.execute(f"INSERT INTO schedule (channel_id, start_date, end_date) VALUES (?, ?, ?);", (chan_id, sched_start_date, sched_start_date))

    conn.commit(); 
    conn.close()
    print('schedules add complete')

def add_programs(programs):
    add_counter = 0
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    channels = get_channels(conn)
    existing_sd_program_ids = get_program_sd_programids(conn)
    for program in programs:
      pgm_ins_str = "INSERT INTO program (sd_programid, title, description, genres, original_air_date, season, episode) values (?,?,?,?,?,?,?);"

      sd_program_id = program.get('programID')
      if sd_program_id in existing_sd_program_ids:
        continue
      existing_sd_program_ids.append(sd_program_id)
      title = ''
      if program.get('titles'):
        first = program.get('titles')[0]
        title = first.get('title120')
      description = ''
      if program.get('descriptions'):
        if program.get('descriptions').get('description1000'):
          first = program.get('descriptions').get('description1000')[0]
          description = first.get('description')
        elif program.get('descriptions').get('description100'):
          first = program.get('descriptions').get('description100')[0]
          description = first.get('description')
      genres = ''
      if program.get('genres'):
        genres = ','.join(program.get('genres'))
      oad = program.get('originalAirDate')
      season = ''
      episode = ''
      for md in program.get('metadata', []):
        if md.get('TVmaze'):
          season = md.get('TVmaze').get('season')
          episode = md.get('TVmaze').get('episode')
        elif md.get('Gracenote'):
          season = md.get('Gracenote').get('season')
          episode = md.get('Gracenote').get('episode')
#      print(f'adding program ===== {sd_program_id} - {title}')
      if not title:
        continue # assume empty record
      cursor.execute(pgm_ins_str, (sd_program_id, title, description, genres, oad, season, episode))
      add_counter += 1
    conn.commit(); 
    conn.close()
    print(f'{len(programs)} programs processed, {add_counter} added')

def add_persons(persons):
    conn = sqlite3.connect(DB_PATH)
#INSERT INTO person (program_id, role, name, character_name) VALUES (1, 'actor', 'Mark Williams', 'Father Brown');
    conn.commit(); 
    conn.close()
    print('persons add complete')

def get_token():
    resp = requests.post(f"{SD_URL}/token", json={'username': SD_USER, 'password': SD_PASS})
    if not resp.ok:
      raise Exception('get token call failed')
    token = resp.json().get('token')
    if not token: 
      raise Exception('cannot get token from schedule direct')
    return token 

def gather_schedule():
    token = get_token()

    status_url = f"{SD_URL}/status"
    status_headers = {'token': token}
    resp = requests.get(status_url, headers=status_headers)
    if not resp.ok:
      raise Exception('cannot get status')
    lineups = resp.json().get('lineups', [])
    if not lineups:
      raise Exception('no lineups found for sd account')
    lineup_id = lineups[0].get('lineup')
    
    channels_url = f"{SD_URL}/lineups/{lineup_id}"
    resp = requests.get(channels_url, headers=status_headers)
    if not resp.ok:
      raise Exception('cannot get channels')
    channels = resp.json().get('map', [])
    if not channels:
      raise Exception('empty channels list found for sd account')
 
    schedule_url = f"{SD_URL}/schedules"
    start_date = datetime.now().strftime('%Y-%m-%d') # TODO check local db, don't get what we already have
    end_date = (datetime.now() + timedelta(days=DAYS_TO_FETCH)).strftime('%Y-%m-%d')
    request_obj = []
    for channel in channels:
      request_obj.append({'stationID': channel.get('stationID'), 'date': [start_date, end_date]})
    resp = requests.post(f"{SD_URL}/schedules", json=request_obj, headers=status_headers)
    if not resp.ok:
      raise Exception('cannot get schedule')
    schedules = resp.json()

    program_ids = set()
    for station in schedules:
      for program in station.get('programs', []):
        program_ids.add(program.get('programID'))     
    program_ids = list(program_ids)
    print(f'gilly [{len(program_ids)}]')
    
    # batch schedule requests in less than 5000 units
    program_url = f"{SD_URL}/programs"
    programs = []
    for i in range(0, len(program_ids), 4500):
        one_batch = program_ids[i:i + 4500]
        resp = requests.post(f"{SD_URL}/programs", json=one_batch, headers=status_headers)
        if not resp.ok:
          raise Exception('cannot get programs, starting at index {i}')
        program_batch = resp.json()
        programs += program_batch  # this will get to be pretty big, batch it if it crashes

    print(f'fetched [{len(programs)}] programs')





#channel - id, station_id, channel
#  schedule  - id, channel_id, start_date, end_date
#    program - id, schedule_id, sd_programid, title, description, genres, original_air_date, season, episode
#      person - id, program_id, role, name, character_name, sd_name_id

# NOW PERSIST THEM
    add_channels(channels)
#    add_schedules(schedules)
    add_programs(programs)

#    add_persons(persons)

    







    return
















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
    parser.add_argument('--mode', default='schedule', choices=['schedule', 'test-epg', '1', '2', '3', '4'])
    args = parser.parse_args()

    if args.mode == '1':
        print(f"=+=+=+=+=+=+=+=+ gathering schedule =+=+=+=+=+=+=+=+")
        init_db()
        schedule = gather_schedule()

    if args.mode == 'schedule':
        print(f"=+=+=+=+=+=+=+=+ Filtering EPG events starting in the next hour based on this prompt: {CUSTOM_PROMPT} =+=+=+=+=+=+=+=+")
        summary_obj = {'prompt': CUSTOM_PROMPT, 'shows': []}
        events_checked = len(events)
        scheduled = 0
        all_show_objects = []
        events = []
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
      print("No future events found")
