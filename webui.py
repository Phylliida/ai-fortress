"""
Flask web UI for worldRefactored.py — a persistent world-builder.

Worlds are stored append-only (store.py: worlds/<id>.jsonl). Flow: create a world (prompt) ->
generate a seed character -> generate locations (ever-filter, keep top 5) -> generate
characters AT each location. Per character: sleep spot / 24h presence grid / schedule.

Needs the GLM server (worldRefactored.SERVER_URL) and the embedding server (EMBED_URL):
  llama-server -m embeddinggemma-300M-Q8_0.gguf --embedding --pooling mean --port 8062
Run:  pip install flask && python3 webui.py  -> http://127.0.0.1:5005
"""
import json
import datetime
from types import SimpleNamespace
from flask import Flask, render_template, request, Response, stream_with_context

import worldRefactored as wr
import store

app = Flask(__name__)
SERVER = wr.LlamaServer()


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def char_obj(rec):
    """Reconstruct a lightweight character (name + prefix + gender/species) from a stored record."""
    return SimpleNamespace(name=rec["name"], gender=rec.get("gender", "female"),
                           species=rec.get("species", "human"), prefix=wr.char_prefix(rec))


def _world_char(wid, cid):
    w = store.load_world(wid)
    if not w:
        return None, None, "World not found."
    rec = store.get_character(wid, cid)
    if not rec:
        return None, None, "Character not found."
    return w, rec, None


@app.route("/")
def index():
    return render_template("index.html", server_url=wr.SERVER_URL)


# --------------------------------------------------------------------------- worlds
@app.route("/api/world", methods=["POST"])
def api_new_world():
    prompt = (request.json or {}).get("prompt", "").strip() or "A fantasy world"
    return {"id": store.new_world(prompt)}


@app.route("/api/worlds")
def api_worlds():
    return {"worlds": store.list_worlds()}


@app.route("/api/world/<wid>")
def api_world(wid):
    w = store.load_world(wid)
    return w if w else ({"error": "not found"}, 404)


# ------------------------------------------------------ character generation (seed / at a place)
@app.route("/api/world/<wid>/character")
def api_world_character(wid):
    w = store.load_world(wid)
    gender = request.args.get("gender", "female").strip()
    species = request.args.get("species", "human").strip()
    at = request.args.get("at", "").strip() or None

    @stream_with_context
    def gen():
        if not w:
            yield sse({"type": "error", "message": "World not found."})
            return
        try:
            setting = w["prompt"] + (f" — at {at}" if at else "")
            ctx = {"world": setting, "species": species, "gender": gender}
            prompts = {"name": wr.field_prompt("name", ctx)}
            ctx["name"] = nm = wr.gen_field(SERVER, "name", ctx)
            yield sse({"type": "field", "field": "name", "value": nm, "prompt": prompts["name"]})
            for field in ("appearance", "personality", "backstory"):
                prompts[field] = wr.field_prompt(field, ctx)
                ctx[field] = val = wr.gen_field(SERVER, field, ctx)
                yield sse({"type": "field", "field": field, "value": val, "prompt": prompts[field]})
            char = SimpleNamespace(name=nm, gender=gender, species=species, prefix=wr.char_prefix(ctx))
            prompts["nocturnal"] = wr.nocturnal_prompt(char)
            noc = wr.nocturnal_prob(SERVER, char)
            yield sse({"type": "nocturnal", "value": noc, "prompt": prompts["nocturnal"]})
            cid = store.new_id(8)
            store.append(wid, {"type": "character", "cid": cid, "role": "location" if at else "seed",
                               "at": at, "world": setting, "species": species, "gender": gender,
                               "name": nm, "appearance": ctx["appearance"],
                               "personality": ctx["personality"], "backstory": ctx["backstory"],
                               "nocturnal": noc, "prompts": prompts})
            yield sse({"type": "saved", "cid": cid})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


# ------------------------------------------------- locations: generate, ever-filter, keep top 5
@app.route("/api/world/<wid>/locations")
def api_world_locations(wid):
    w = store.load_world(wid)
    n = int(request.args.get("n", "12"))
    top = int(request.args.get("top", "5"))

    @stream_with_context
    def gen():
        if not w:
            yield sse({"type": "error", "message": "World not found."})
            return
        seeds = [c for c in w["characters"] if c.get("role") == "seed"]
        if not seeds:
            yield sse({"type": "error", "message": "Generate a seed character first."})
            return
        char = char_obj(seeds[0])
        try:
            cands = list(wr.iter_locations(SERVER, char, n=n))
            scored = []
            for loc in cands:                              # ever-filter -> keep top `top`
                q = wr.ever_prompt(char, loc)
                p = SERVER.yes_no_prob(q)
                scored.append((loc, p))
                yield sse({"type": "ever", "location": loc, "prob": p, "prompt": q})
            kept = [loc for loc, _ in sorted(scored, key=lambda x: -x[1])[:top]]
            store.append(wid, {"type": "locations", "kept": kept})
            yield sse({"type": "kept", "locations": kept})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


# --------------------------------------------------- per-character analysis (uses world locations)
@app.route("/api/world/<wid>/char/<cid>/sleep")
def api_char_sleep(wid, cid):
    w, rec, err = _world_char(wid, cid)

    @stream_with_context
    def gen():
        if err:
            yield sse({"type": "error", "message": err}); return
        if not w["locations"]:
            yield sse({"type": "error", "message": "Generate locations first."}); return
        char = char_obj(rec)
        try:
            for loc in w["locations"]:
                q = wr.sleep_prompt(char, loc)
                yield sse({"type": "sleep", "location": loc, "prob": SERVER.yes_no_prob(q), "prompt": q})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/world/<wid>/char/<cid>/presence")
def api_char_presence(wid, cid):
    w, rec, err = _world_char(wid, cid)

    @stream_with_context
    def gen():
        if err:
            yield sse({"type": "error", "message": err}); return
        if not w["locations"]:
            yield sse({"type": "error", "message": "Generate locations first."}); return
        char = char_obj(rec)
        try:
            times = wr.make_times(datetime.datetime.strptime("00:00", "%H:%M"), 23, hours=1)
            for t in times:                                # all 24 hours over the 5 world locations
                ts = t.strftime("%H:%M")
                for loc in w["locations"]:
                    q = wr.presence_prompt(char, loc, ts)
                    yield sse({"type": "presence", "time": ts, "location": loc,
                               "prob": SERVER.yes_no_prob(q), "prompt": q})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/world/<wid>/char/<cid>/schedule")
def api_char_schedule(wid, cid):
    w, rec, err = _world_char(wid, cid)
    start = request.args.get("start", "08:00").strip()
    steps = int(request.args.get("steps", "12"))

    @stream_with_context
    def gen():
        if err:
            yield sse({"type": "error", "message": err}); return
        char = char_obj(rec)
        try:
            locations = w["locations"] or wr.gen_locations(SERVER, char)
            times = wr.make_times(datetime.datetime.strptime(start, "%H:%M"), steps, hours=1)
            durs = [f"{a.strftime('%I:%M %p')}-{b.strftime('%I:%M %p')}"
                    for a, b in zip(times, times[1:])]
            prompt = char.prefix + f"\n\nThis is what {char.name} did yesterday:\n"
            for dur in durs:
                ap = prompt + dur + ":"
                action = SERVER.gen_text(ap, stop=["\n", "|", "("], n_predict=40)
                lp = prompt + dur + ": " + action + ". Location:"
                loc = SERVER.pick_from_set(lp, locations)
                yield sse({"type": "slot", "time": dur, "action": action, "location": loc,
                           "action_prompt": ap, "loc_prompt": lp})
                prompt = prompt + dur + ": " + action + " | Location: " + loc + "\n"
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=True, threaded=True)
