"""SKYFORGE JSON link protocol (F9/F10).

Envelope (every message, < 1 KB asserted in code):
    {"v":1, "seq":1832, "ts":1762560000.12, "b":4711, "t":"det", ...body..., "h":"9f2a1c7e4b0d3a11"}

  v    protocol version         seq  per-sender sequence number
  ts   sender clock (s)         b    sender boot id (seq restarts on reboot)
  t    type: tel det dec goal ack sync may frag
  h    HMAC-SHA256 over the canonical JSON (sorted keys, no "h"), first 8 bytes hex

Messages that do not fit (MAYDAY / sync snapshots) are split into "frag"
messages: {"t":"frag","ft":<type>,"fid":<group>,"fi":i,"fn":n,"d":"<json slice>"}.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Optional

PRIORITY = {"tel": 0, "det": 1, "dec": 1, "goal": 2, "ack": 2, "sync": 2, "may": 2}
ENVELOPE = ("v", "seq", "ts", "b", "t", "h")


class ForgedMessage(Exception):
    pass


class MessageTooLarge(Exception):
    pass


def _canonical(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")


def _compact(msg: dict) -> bytes:
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class Codec:
    def __init__(self, key: bytes, version: int = 1, max_bytes: int = 1024, boot_id: int = 0):
        self.key = key
        self.version = version
        self.max_bytes = max_bytes
        self.boot_id = boot_id

    @classmethod
    def from_params(cls, params: dict, boot_id: int = 0) -> "Codec":
        c = params["comm"]
        return cls(bytes.fromhex(c["key_hex"]), int(c["version"]), int(c["max_msg_bytes"]), boot_id)

    def tag(self, msg: dict) -> str:
        return hmac.new(self.key, _canonical(msg), hashlib.sha256).hexdigest()[:16]

    def encode(self, t: str, body: dict, seq: int, ts: float) -> bytes:
        msg: dict[str, Any] = {"v": self.version, "seq": seq, "ts": round(ts, 2), "b": self.boot_id, "t": t}
        for k, v in body.items():
            if k in ENVELOPE:
                raise ValueError(f"body key {k!r} collides with the envelope")
            msg[k] = v
        msg["h"] = self.tag(msg)
        out = _compact(msg)
        if len(out) >= self.max_bytes:
            raise MessageTooLarge(f"{t} message is {len(out)} bytes (limit {self.max_bytes})")
        return out

    def decode(self, data: bytes) -> dict:
        try:
            msg = json.loads(data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data)
        except (ValueError, UnicodeDecodeError) as e:
            raise ForgedMessage(f"malformed: {e}") from e
        if not isinstance(msg, dict) or "h" not in msg:
            raise ForgedMessage("missing tag")
        h = msg.pop("h")
        if not hmac.compare_digest(str(h), self.tag(msg)):
            raise ForgedMessage("bad HMAC")
        if msg.get("v") != self.version:
            raise ForgedMessage(f"version {msg.get('v')}")
        return msg

    # ---------------------------------------------------------------- frags
    def fits(self, t: str, body: dict) -> bool:
        try:
            self.encode(t, body, 2**31, 9.99e9)
            return True
        except MessageTooLarge:
            return False

    def fragment(self, t: str, body: dict, fid: int) -> list[dict]:
        """Split a body that does not fit into frag bodies (each fits)."""
        raw = json.dumps(body, separators=(",", ":"), ensure_ascii=True)
        overhead = len(self.encode("frag", {"ft": t, "fid": fid, "fi": 999, "fn": 999, "d": ""}, 2**31, 9.99e9))
        chunk = self.max_bytes - overhead - 32  # headroom for json escaping of quotes
        parts = []
        i = 0
        while i < len(raw):
            # escaping inflates some characters; shrink the slice until it fits
            n = chunk
            while True:
                piece = raw[i : i + n]
                body_f = {"ft": t, "fid": fid, "fi": len(parts), "fn": 0, "d": piece}
                if self.fits("frag", body_f):
                    break
                n = int(n * 0.8)
            parts.append(piece)
            i += n
        return [{"ft": t, "fid": fid, "fi": k, "fn": len(parts), "d": p} for k, p in enumerate(parts)]


class Reassembler:
    def __init__(self):
        self.groups: dict[tuple, dict[int, str]] = {}

    def add(self, sender_b: int, msg: dict) -> Optional[tuple[str, dict]]:
        key = (sender_b, msg["fid"])
        g = self.groups.setdefault(key, {})
        g[msg["fi"]] = msg["d"]
        if len(g) == msg["fn"]:
            raw = "".join(g[i] for i in range(msg["fn"]))
            del self.groups[key]
            return msg["ft"], json.loads(raw)
        return None


class ReplayGuard:
    """Accept only strictly newer seq per sender session (boot id)."""

    def __init__(self):
        self.last: dict[int, int] = {}
        self.last_ts = -1e18
        self.last_b: Optional[int] = None

    def check(self, msg: dict) -> bool:
        b, seq, ts = msg.get("b"), msg.get("seq"), float(msg.get("ts", 0))
        if b == self.last_b:
            ok = seq > self.last.get(b, -1)
        else:
            ok = ts > self.last_ts  # new session must also be newer in time
        if ok:
            self.last[b] = seq
            self.last_b = b
            self.last_ts = max(self.last_ts, ts)
        return ok
