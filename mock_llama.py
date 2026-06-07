"""Tiny stand-in for llama-server's /completion, to test the UI wiring without
a real model. Returns canned prose for free generation, and (when a grammar is
present) the grammar's first quoted literal — so pick_from_set/yes_no behave."""
import json, re, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

CANNED = ["explored the damp tunnels", "nibbled on glowing fungus",
          "rested in the warm silt", "argued with a passing beetle",
          "polished a favorite pebble", "charted the spore currents"]
_i = [0]

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        grammar = body.get("grammar", "")
        if grammar:
            lits = re.findall(r'"([^"]+)"', grammar)
            content = lits[0] if lits else "Yes"
        else:
            content = CANNED[_i[0] % len(CANNED)]; _i[0] += 1
        resp = json.dumps({
            "content": content,
            "completion_probabilities": [{"probs": [{"tok": "Yes", "prob": 0.72},
                                                     {"tok": "No", "prob": 0.28}]}],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a):
        pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    HTTPServer(("127.0.0.1", port), H).serve_forever()
