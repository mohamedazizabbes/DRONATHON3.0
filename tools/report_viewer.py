import argparse
import glob
import os
import shutil
import subprocess
import time


def browser_cmd(path):
    url = "file://" + os.path.abspath(path)
    for b in ("google-chrome-stable", "google-chrome", "chromium"):
        if shutil.which(b):
            return [b, "--kiosk", "--new-window", "--no-first-run", url]
    if shutil.which("firefox"):
        return ["firefox", "--kiosk", url]
    return ["xdg-open", url]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs"))
    ap.add_argument("--interval", type=float, default=2.0)
    a = ap.parse_args()
    seen = set(glob.glob(os.path.join(a.dir, "mission_report_*.html")))
    print(f"watching {os.path.abspath(a.dir)} for new mission reports")
    while True:
        for f in sorted(glob.glob(os.path.join(a.dir, "mission_report_*.html")), key=os.path.getmtime):
            if f not in seen:
                seen.add(f)
                time.sleep(1.0)
                print(f"opening {f}")
                subprocess.Popen(browser_cmd(f), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
