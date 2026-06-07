"""
Minimal Flask web UI for worldRefactored.py — drive the base-model world generator from a
browser. Generation is slow (many sequential model calls), so results STREAM over
Server-Sent Events: character fields, locations, and schedule slots appear live.

Flow: generate a character -> generate the locations it frequents -> generate its schedule
(which picks from those locations).

Needs two servers up:
  * the GLM base model   (worldRefactored.SERVER_URL, e.g. 172.22.146.1:8055)
  * the embedding model  (worldRefactored.EMBED_URL, local :8062) for location dedup
      llama-server -m embeddinggemma-300M-Q8_0.gguf --embedding --pooling mean --port 8062

Run:  pip install flask && python3 webui.py  -> http://127.0.0.1:5005
"""
import json
import datetime
from types import SimpleNamespace
from flask import Flask, render_template, request, Response, stream_with_context

import worldRefactored as wr

app = Flask(__name__)
SERVER = wr.LlamaServer()
STATE = {"character": None, "locations": []}   # single-user local demo


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


@app.route("/")
def index():
    return render_template("index.html", server_url=wr.SERVER_URL)


@app.route("/api/character")
def api_character():
    world = request.args.get("world", "A medieval fantasy world").strip()
    gender = request.args.get("gender", "female").strip()
    species = request.args.get("species", "human").strip()
    name = request.args.get("name", "").strip() or None

    @stream_with_context
    def gen():
        try:
            ctx = {"world": world, "species": species, "gender": gender}     # few-shot primed
            nprompt = wr.field_prompt("name", ctx)
            ctx["name"] = nm = name or wr.gen_field(SERVER, "name", ctx)
            yield sse({"type": "field", "field": "name", "value": nm,
                       "prompt": "(user-provided name)" if name else nprompt})
            for field in ("appearance", "personality", "backstory"):
                fprompt = wr.field_prompt(field, ctx)
                ctx[field] = val = wr.gen_field(SERVER, field, ctx)
                yield sse({"type": "field", "field": field, "value": val, "prompt": fprompt})
            char = SimpleNamespace(world=world, species=species, gender=gender,
                                   name=nm, prefix=wr.char_prefix(ctx))
            STATE["character"] = char
            STATE["locations"] = []
            yield sse({"type": "nocturnal", "value": wr.nocturnal_prob(SERVER, char),
                       "prompt": wr.nocturnal_prompt(char)})   # separate character-level ask
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/locations")
def api_locations():
    char = STATE["character"]
    n = int(request.args.get("n", "12"))

    @stream_with_context
    def gen():
        if char is None:
            yield sse({"type": "error", "message": "Generate a character first."})
            return
        try:
            locs = []
            lp = wr.locations_prompt(char)
            for loc in wr.iter_locations(SERVER, char, n=n):
                locs.append(loc)
                yield sse({"type": "location", "value": loc, "prompt": lp})
            STATE["locations"] = locs
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/sleep")
def api_sleep():
    char = STATE["character"]

    @stream_with_context
    def gen():
        if char is None:
            yield sse({"type": "error", "message": "Generate a character first."})
            return
        locs = STATE.get("locations") or []
        if not locs:
            yield sse({"type": "error", "message": "Generate locations first."})
            return
        try:
            for loc in locs:
                q = wr.sleep_prompt(char, loc)
                yield sse({"type": "sleep", "location": loc, "prob": SERVER.yes_no_prob(q), "prompt": q})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/presence")
def api_presence():
    char = STATE["character"]
    top = int(request.args.get("top", "5"))

    @stream_with_context
    def gen():
        if char is None:
            yield sse({"type": "error", "message": "Generate a character first."})
            return
        locs = STATE.get("locations") or []
        if not locs:
            yield sse({"type": "error", "message": "Generate locations first."})
            return
        try:
            # phase 1: cheap "ever there?" pass — one call per location, keep the top `top`
            scored = []
            for loc in locs:
                q = wr.ever_prompt(char, loc)
                p = SERVER.yes_no_prob(q)
                scored.append((loc, p))
                yield sse({"type": "ever", "location": loc, "prob": p, "prompt": q})
            kept = [loc for loc, _ in sorted(scored, key=lambda x: -x[1])[:top]]
            yield sse({"type": "kept", "locations": kept})
            # phase 2: hourly presence over ALL 24 hours (00:00–23:00), kept locations only
            times = wr.make_times(datetime.datetime.strptime("00:00", "%H:%M"), 23, hours=1)
            for t in times:
                ts = t.strftime("%H:%M")
                for loc in kept:
                    q = wr.presence_prompt(char, loc, ts)
                    yield sse({"type": "presence", "time": ts, "location": loc,
                               "prob": SERVER.yes_no_prob(q), "prompt": q})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/schedule")
def api_schedule():
    char = STATE["character"]
    start = request.args.get("start", "08:00").strip()
    steps = int(request.args.get("steps", "12"))

    @stream_with_context
    def gen():
        if char is None:
            yield sse({"type": "error", "message": "Generate a character first."})
            return
        try:
            locations = STATE.get("locations") or wr.gen_locations(SERVER, char)
            times = wr.make_times(datetime.datetime.strptime(start, "%H:%M"), steps, hours=1)
            durs = [f"{a.strftime('%I:%M %p')}-{b.strftime('%I:%M %p')}"
                    for a, b in zip(times, times[1:])]
            prompt = char.prefix + f"\n\nThis is what {char.name} did yesterday:\n"
            for dur in durs:
                action_prompt = prompt + dur + ":"
                action = SERVER.gen_text(action_prompt, stop=["\n", "|", "("], n_predict=40)
                loc_prompt = prompt + dur + ": " + action + ". Location:"
                loc = SERVER.pick_from_set(loc_prompt, locations)
                yield sse({"type": "slot", "time": dur, "action": action, "location": loc,
                           "action_prompt": action_prompt, "loc_prompt": loc_prompt})
                prompt = prompt + dur + ": " + action + " | Location: " + loc + "\n"
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


if __name__ == "__main__":
    # threaded=True so the SSE stream doesn't block other requests
    app.run(host="127.0.0.1", port=5005, debug=True, threaded=True)
