import argparse
import pandas as pd
import subprocess
import time
import os
import signal
from typing import Dict

# Reuse test function from first script (adapt as needed)
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

# VPN handling functions
def start_vpn(ovpn_file: str) -> subprocess.Popen:
    if not os.path.exists(ovpn_file):
        raise FileNotFoundError(f"OVPN file not found: {ovpn_file}")
    print(f"Starting VPN with {ovpn_file}...")
    proc = subprocess.Popen(["sudo", "openvpn", "--config", ovpn_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Wait for connection (poll logs for "Initialization Sequence Completed")
    start_time = time.time()
    while time.time() - start_time < 60:  # Timeout after 60s
        line = proc.stdout.readline().decode().strip()
        if line:
            print(line)
        if "Initialization Sequence Completed" in line:
            print("VPN connected!")
            return proc
        time.sleep(1)
    raise TimeoutError("VPN connection timed out")

def stop_vpn(proc: subprocess.Popen):
    print("Stopping VPN...")
    os.kill(proc.pid, signal.SIGTERM)
    proc.wait()
    print("VPN stopped.")

# Main
def main():
    parser = argparse.ArgumentParser(description="Refine IPTV CSV by retrying failed streams, optionally with VPN.")
    parser.add_argument("--input_csv", default="working_iptv_channels.csv", help="Input CSV from harvest script")
    parser.add_argument("--group", required=True, help="Group to filter (e.g., 'spain')")
    parser.add_argument("--vpn_country", help="VPN country code (e.g., 'es' for Spain; requires .ovpn file)")
    parser.add_argument("--ovpn_dir", default="~/nordvpn/", help="Directory with .ovpn files (e.g., ~/nordvpn/)")
    parser.add_argument("--output_csv", default="refined_iptv_channels.csv", help="Output updated CSV")

    args = parser.parse_args()

    # Load CSV
    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} channels from {args.input_csv}")

    # Filter: failed (working == False) and specific group
    mask = (df["working"] == False) & (df["group"].str.lower() == args.group.lower())
    to_retry = df[mask].copy()
    print(f"Found {len(to_retry)} failed channels in group '{args.group}' to retry")

    if len(to_retry) == 0:
        print("No channels to retry. Exiting.")
        return

    vpn_proc = None
    try:
        if args.vpn_country:
            ovpn_file = os.path.expanduser(os.path.join(args.ovpn_dir, f"{args.vpn_country}.ovpn"))
            vpn_proc = start_vpn(ovpn_file)
            time.sleep(5)  # Extra settle time

        # Retry tests
        for idx, row in to_retry.iterrows():
            print(f"Retrying {row['name']} ({row['url']})...")
            success = test_stream(row.to_dict())
            if success:
                print("  - Success with retry!")
                df.loc[idx, "working"] = True
                if args.vpn_country:
                    df.loc[idx, "notes"] = f"{row['notes']} (Requires VPN: {args.vpn_country})"
            else:
                print("  - Still failed")
            time.sleep(2)  # Rate limit

    finally:
        if vpn_proc:
            stop_vpn(vpn_proc)

    # Save updated
    df.to_csv(args.output_csv, index=False)
    print(f"Saved updated CSV to {args.output_csv}")

if __name__ == "__main__":
    main()
