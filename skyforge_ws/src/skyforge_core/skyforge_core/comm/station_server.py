"""Ground station process: UDP link to the drone + tiny HTTP API for the operator.

    GET  /status              last telemetry, detections, decisions, link stats
    POST /goal     {json}     signed goal_update, e.g. {"type":"goto","p":[120,40]}
    POST /forge    {json}     demo: goal_update signed with a WRONG key (must be rejected)
    POST /replay              demo: resend the last real goal_update (must be rejected)

Run:  python -m skyforge_core.comm.station_server --config skyforge.yaml
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..params import load_params
from .station import GroundStation


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SKYFORGE ground station")
    ap.add_argument("--config", default=None)
    ap.add_argument("--drone", default=None, help="drone comm host:port (default from config)")
    ap.add_argument("--log", default="station_log.jsonl")
    a = ap.parse_args(argv)
    params = load_params(a.config)
    udp = params["comm"]["udp"]
    drone_addr = (udp["drone_host"], int(udp["drone_port"]))
    if a.drone:
        h, p = a.drone.rsplit(":", 1)
        drone_addr = (h, int(p))

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", int(udp["station_port"])))
    lock = threading.Lock()

    def tx(data: bytes) -> bool:
        sock.sendto(data, drone_addr)
        return True

    def on_mayday(msg: dict) -> None:
        with open("mayday_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(msg, f, indent=1)
        print("[station] MAYDAY -> snapshot saved to mayday_snapshot.json, dispatch drone 2", flush=True)

    gs = GroundStation(params, tx, on_mayday)
    logf = open(a.log, "a", encoding="utf-8")

    def rx_loop():
        sock.settimeout(0.2)
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                data = None
            with lock:
                if data:
                    gs.on_receive(data, time.time())
                    logf.write(data.decode("utf-8", "replace") + "\n")
                    logf.flush()
                gs.tick(time.time())

    threading.Thread(target=rx_loop, daemon=True).start()

    class H(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            with lock:
                self._json(200, {
                    "tel": gs.last_tel, "detections": gs.detections, "decisions": gs.decisions[-30:],
                    "contig_seq": gs.contig, "max_seq": gs.max_seen, "stats": gs.stats,
                    "resync_s": gs.resync_times[-5:], "mayday": gs.mayday is not None,
                })

        def do_POST(self):  # noqa: N802
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            with lock:
                if self.path == "/goal":
                    gs.send_goal(body, time.time())
                elif self.path == "/forge":
                    gs.send_forged_goal(body or {"type": "rtb"}, time.time())
                elif self.path == "/replay":
                    gs.replay_last_goal()
                else:
                    return self._json(404, {"error": "unknown"})
            self._json(200, {"ok": True})

        def log_message(self, *args):
            pass

    port = int(params["comm"]["http_port"])
    print(f"[station] UDP :{udp['station_port']} -> drone {drone_addr}; HTTP :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
