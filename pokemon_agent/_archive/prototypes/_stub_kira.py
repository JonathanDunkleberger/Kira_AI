"""_stub_kira.py - a CAPTURE STUB standing in for the live Kira control server.

It speaks just enough of kira/dashboard/control_server.py's contract to PROVE the
soul-flow wiring end-to-end when the real bot isn't up:
  POST /cmd/pokemon_event          -> logs the summary that would reach _pokemon_react
  GET  /state                      -> {"is_speaking": false}  (so pacing is a no-op)
  POST /cmd/pokemon_choose_starter -> a canned pick

Every received event is printed AND appended to a JSONL file so the exact ordered
stream that reached her seam can be read back. This is NOT her voice - it's the proof
that the events FLOW. Her real lines come from the live bot (same POST contract).

Run:  .venv\\Scripts\\python.exe pokemon_agent\\_stub_kira.py [--port 8766] [--out events.jsonl]
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

OUT = None
N = [0]


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # silence the default per-request stderr spam

    def do_GET(self):
        if self.path.rstrip("/") == "/state":
            return self._send({"is_speaking": False})
        self._send({"ok": True, "path": self.path})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw) if raw else {}
        except Exception:
            body = {"_raw": raw.decode("utf-8", "replace")}
        path = self.path.rstrip("/")
        if path == "/cmd/pokemon_event":
            summary = (body.get("name") or body.get("text") or "").strip()
            tier = body.get("tier")
            N[0] += 1
            rec = {"i": N[0], "t": round(time.time(), 2), "tier": tier, "summary": summary}
            print(f"   [stub-kira] EVENT #{N[0]:>3} T{tier}: {summary!r}", flush=True)
            if OUT:
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            return self._send({"ok": True, "fired": bool(summary)})
        if path == "/cmd/pokemon_choose_starter":
            print("   [stub-kira] choose_starter -> bulbasaur", flush=True)
            return self._send({"ok": True, "choice": "bulbasaur", "reasoning": "(stub)"})
        self._send({"ok": True, "path": path})


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--out", default=None, help="append received events as JSONL")
    args = ap.parse_args()
    OUT = args.out
    if OUT:
        open(OUT, "w").close()  # truncate
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"   [stub-kira] capturing on :{args.port}  out={OUT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
