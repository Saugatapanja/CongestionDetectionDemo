import asyncio
import logging
import os
import sys
import threading
import time
import webbrowser
import uvicorn

# Suppress benign Windows socket shutdown exceptions (WinError 10054) on browser reload
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

from tools.sample_video_generator import generate_traffic_sample, generate_crowd_sample


def ensure_samples():
    traffic_sample = os.path.join("data", "sample_videos", "traffic_demo.mp4")
    crowd_sample = os.path.join("data", "sample_videos", "crowd_demo.mp4")
    if not os.path.exists(traffic_sample):
        print("[INIT] Generating traffic simulation sample video...")
        generate_traffic_sample(traffic_sample)
    if not os.path.exists(crowd_sample):
        print("[INIT] Generating pandal crowd simulation sample video...")
        generate_crowd_sample(crowd_sample)


def open_browser(url: str):
    time.sleep(1.8)
    print(f"\n[DASHBOARD] Opening Web Command Center: {url}\n")
    webbrowser.open(url)


def main():
    print("\n" + "=" * 65)
    print("  AI POLICE COMMAND CENTER - TRAFFIC & PUJA CROWD MANAGEMENT")
    print("=" * 65)

    ensure_samples()

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print(f"\n[SERVER] Starting FastAPI Dashboard on {url}")
    print("Press Ctrl+C to stop the dashboard server.\n")

    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    uvicorn.run("web.app:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
