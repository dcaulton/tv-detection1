import requests
import pandas as pd
import subprocess
import time
from typing import List, Dict



# Hardcoded list of .md files from repo (filtered to key ones; add more from full list if needed)
md_files = [
    "argentina.md", "australia.md", "austria.md", "belgium.md", "brazil.md", "canada.md", "chile.md",
    "france.md", "germany.md", "india.md", "italy.md", "japan.md", "mexico.md", "netherlands.md",
    "spain.md", "sweden.md", "switzerland.md", "uk.md", "usa.md", "zz_news_en.md", "zz_news_es.md"
    # Add more like "russia.md" if wanted; full list is ~80
]

base_raw_url = "https://raw.githubusercontent.com/Free-TV/IPTV/master/lists/"

def fetch_md_content(filename: str) -> str:
    url = base_raw_url + filename
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to fetch {filename}")
        return ""

def parse_channels(content: str, group: str) -> List[Dict]:
    channels = []
    lines = content.splitlines()
    first_two = lines[0:9];
    print(f'-------\n---------parsing new channel, starts like this: {first_two}')
    in_table = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| # ") or stripped.startswith("|:--"):  # Header or separator
            in_table = True
            continue
        if in_table and stripped.startswith("|") and len(stripped.split("|")) >= 4:
            parts = [p.strip() for p in stripped.split("|")[1:-1]]  # Trim outer empties
            if len(parts) < 3:
                continue
            name_raw = parts[0]
            if "[x]" in name_raw.lower():  # Skip invalid/dead
                continue

            # URL extraction: look for first markdown link [text](url)
            url = ""
            for part in parts[1:]:
                if "](" in part:
                    start = part.find("](") + 2
                    end = part.rfind(")")
                    if start > 1 and end > start:
                        url = part[start:end].strip()
                        break
            if not url or not url.startswith("http"):
                continue

            notes = " ".join(parts[2:]) if len(parts) > 2 else ""

            geo_blocked = "Ⓖ" in name_raw or "Ⓖ" in notes
            is_youtube = "Ⓨ" in name_raw or "Ⓨ" in notes or "youtube.com" in url.lower() or "youtu.be" in url.lower()
            channel_type = "youtube" if is_youtube else ("hls" if any(ext in url.lower() for ext in [".m3u", ".m3u8", ".smil"]) else "other")

            clean_name = name_raw.replace("[x]", "").replace("[>]", "").strip()

            channels.append({
                "group": group.replace(".md", ""),
                "name": clean_name,
                "url": url,
                "type": channel_type,
                "notes": notes,
                "geo_blocked": geo_blocked,
                "working": False,
                "vpn_suggestion": group.replace(".md", "") if geo_blocked else ""
            })
    print(f'  parse channels: [{len(channels)}] channels found')
    return channels

def test_stream(channel: Dict) -> bool:
    url = channel["url"]
    cmd = []
    if channel["type"] == "youtube":
        cmd = ["yt-dlp", "--simulate", "-f", "best", url]
    else:  # hls/other
        cmd = ["ffmpeg", "-i", url, "-t", "5", "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        return result.returncode == 0
    except Exception as e:
        print(f"Test failed for {channel['name']}: {e}")
        return False

#####################################################################################33
#url = "https://raw.githubusercontent.com/Free-TV/IPTV/master/lists/chile.md"
#content = requests.get(url).text
#print(content[:200])  # See if it fetched real markdown
#print("Length:", len(content))
#channels = parse_channels(content, "chile.md")
#print(f"Extracted {len(channels)} channels from chile.md")
#if channels:
#    print("First one:", channels[0])
#####################################################################################33
# Main harvest and test
all_channels = []
for filename in md_files:
    print(f"Processing {filename}...")
    content = fetch_md_content(filename)
    if content:
        channels = parse_channels(content, filename)
        all_channels.extend(channels)
    time.sleep(2)  # Avoid GitHub rate limits

# Test (this may take time; ~5-20s per channel)
for i, channel in enumerate(all_channels):
    print(f"Testing {i+1}/{len(all_channels)}: {channel['name']} ({channel['group']})")
    channel["working"] = test_stream(channel)
    if not channel["working"] and channel["geo_blocked"]:
        print(f"  - Failed; try with VPN for {channel['vpn_suggestion']}")
    time.sleep(1)  # Avoid stream provider bans

# Output to CSV
df = pd.DataFrame(all_channels)
df.to_csv("working_iptv_channels.csv", index=False)
print("Done! Check working_iptv_channels.csv for results. Filter 'working' == True for usable ones.")
