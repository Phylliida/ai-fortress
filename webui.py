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
import baseModelPrimitives as bmp
import needs
import sim
import store

app = Flask(__name__)
SERVER = wr.LlamaServer()


@app.before_request
def _route_query_log():
    """Route this request's base-model queries into the targeted world's log (audit/debug)."""
    parts = request.path.strip("/").split("/")          # api / world / <wid> / ...
    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "world":
        wid = parts[2]
        bmp.set_log_sink(lambda e: store.append(wid, {"type": "query", **e}))
    else:
        bmp.set_log_sink(None)
    # NB: no teardown clear — it fires before the SSE generator streams its queries. The next
    # request's before_request resets the sink, so it can't leak to the wrong world.


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


@app.route("/api/world/<wid>/log")
def api_world_log(wid):
    """Chronological base-model query log for a world (prompt + output). since>0 = only newer."""
    return {"entries": store.get_log(wid, int(request.args.get("since", 0)))}


@app.route("/api/world/<wid>/place", methods=["POST"])
def api_place(wid):
    """Drag-drop a character onto the 500x500 map (or move/remove it). Body: {cid, x, y} — x/y in
    [0,500]; pass x or y null to take the character off the map."""
    if not store.load_world(wid):
        return {"error": "not found"}, 404
    d = request.json or {}
    cid = d.get("cid")
    if not cid or not store.get_character(wid, cid):
        return {"error": "character not found"}, 400
    x, y = d.get("x"), d.get("y")
    if x is not None and y is not None:
        x = max(0.0, min(500.0, float(x)))
        y = max(0.0, min(500.0, float(y)))
    else:
        x = y = None
    store.append(wid, {"type": "place", "cid": cid, "x": x, "y": y})
    return {"ok": True, "cid": cid, "x": x, "y": y}


_THRESH_CACHE = {}   # per-need deadband, baked once (bake_thresholds is species-agnostic) and reused


def _thresholds_for(need_list):
    """Per-need deadband fractions for `need_list`, baking only the not-yet-cached ones."""
    missing = [n for n in need_list if n not in _THRESH_CACHE]
    if missing:
        _THRESH_CACHE.update(needs.bake_thresholds(SERVER, missing, samples=5))
    return {n: _THRESH_CACHE[n] for n in need_list}


def _world_species_needs(w):
    """Map EVERY species present in the world (default {human}) to the need-list to test items
    against — the union of that species' characters' needs, else the universal core minus excludes."""
    bysp = {}
    for c in w.get("characters", []):
        bysp.setdefault(c.get("species", "human"), set()).update(c.get("needs", []))
    if not bysp:
        bysp["human"] = set()
    core = [n for n in needs.UNIVERSAL_CORE if n not in needs.EXCLUDE_NEEDS]
    return {sp: (sorted(ns) if ns else core) for sp, ns in bysp.items()}


@app.route("/api/world/<wid>/item")
def api_create_item(wid):
    """Create an item TEMPLATE from a free-text name. Streams which needs it can fill (affordance
    gate + gen_percent refill), then stores it for the palette."""
    w = store.load_world(wid)
    name = request.args.get("name", "").strip()

    @stream_with_context
    def gen():
        if not w:
            yield sse({"type": "error", "message": "World not found."}); return
        if not name:
            yield sse({"type": "error", "message": "No item name."}); return
        try:
            affords = {}                                   # {species: {need: refill}}
            for sp, need_list in _world_species_needs(w).items():
                sp_aff = {}
                for need in need_list:
                    r = needs.bake_affordance(SERVER, sp, name, need)
                    if r["applies"] and r["refill"] > 0:
                        sp_aff[need] = round(r["refill"], 3)
                        yield sse({"type": "afford", "species": sp, "need": need, "refill": sp_aff[need]})
                affords[sp] = sp_aff
            # how long using the item takes (for the sim's gradual refill); named-action via a species
            # that it actually serves. None if it fills nothing.
            # durations are PER NEED (a royal bed: sleep 8h, novelty a few min), so using the item for
            # one need doesn't borrow another's duration.
            served = sorted({n for sp in affords for n in affords[sp]})
            durations, durations_ok = {}, {}
            for need in served:
                yield sse({"type": "status", "message": f"timing the {need} activity…"})
                spn = next(s for s in affords if need in affords[s])
                dr = SERVER.gen_duration(needs.duration_prompt(spn, name, need), samples=8,
                                         check_subject=f"using a {name} to satisfy their {need} need")
                if dr:
                    durations[need] = max(round(dr["minutes"], 1), 0.5)
                    durations_ok[need] = round(dr["p_makes_sense"], 3) if dr.get("p_makes_sense") is not None else None
                yield sse({"type": "duration", "need": need,
                           "duration_min": durations.get(need), "duration_ok": durations_ok.get(need)})
            consumable = SERVER.yes_no_prob(needs.consumable_prompt(name)) >= 0.5   # used up after use?
            yield sse({"type": "consumable", "consumable": consumable})
            iid = store.new_id(8)
            store.append(wid, {"type": "item", "iid": iid, "name": name, "affords": affords,
                               "durations": durations, "durations_ok": durations_ok,
                               "consumable": consumable})
            yield sse({"type": "saved", "iid": iid, "affords": affords})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


@app.route("/api/world/<wid>/item_place", methods=["POST"])
def api_item_place(wid):
    """Place/move/remove an item instance on the map. Body: {iid, x, y, pid?} — omit pid to create a
    new instance (palette stamp); pass pid to move it; pass x/y null to remove it."""
    if not store.load_world(wid):
        return {"error": "not found"}, 404
    d = request.json or {}
    iid, pid = d.get("iid"), d.get("pid") or store.new_id(8)
    x, y = d.get("x"), d.get("y")
    if x is not None and y is not None:
        x = max(0.0, min(500.0, float(x)))
        y = max(0.0, min(500.0, float(y)))
    else:
        x = y = None
    store.append(wid, {"type": "item_place", "pid": pid, "iid": iid, "x": x, "y": y})
    return {"ok": True, "pid": pid, "iid": iid, "x": x, "y": y}


@app.route("/api/world/<wid>/item/<iid>/delete", methods=["POST"])
def api_item_delete(wid, iid):
    """Delete an item template from the palette (and its placed instances)."""
    if not store.load_world(wid):
        return {"error": "not found"}, 404
    store.append(wid, {"type": "item_delete", "iid": iid})
    return {"ok": True}


@app.route("/api/world/<wid>/item/<iid>/duration", methods=["POST"])
def api_item_duration(wid, iid):
    """Manually set an item's PER-NEED activity duration (minutes) — correct a gate-flagged guess."""
    w = store.load_world(wid)
    if not w or iid not in w["items"]:
        return {"error": "not found"}, 404
    d = request.json or {}
    need = d.get("need")
    dm = d.get("duration_min")
    dm = max(0.5, float(dm)) if dm is not None else None
    store.append(wid, {"type": "item_duration", "iid": iid, "need": need, "duration_min": dm})
    return {"ok": True, "need": need, "duration_min": dm}


@app.route("/api/world/<wid>/sim/step", methods=["POST"])
def api_sim_step(wid):
    """Advance the live colony sim by `ticks` ticks and return the new state (agent positions +
    need levels + what each is doing). In-memory; reconciles placed characters/items each call."""
    w = store.load_world(wid)
    if not w:
        return {"error": "not found"}, 404
    ticks = max(1, min(int((request.json or {}).get("ticks", 1)), 240))
    return sim.step_world(w, ticks)


@app.route("/api/world/<wid>/sim/reset", methods=["POST"])
def api_sim_reset(wid):
    sim.reset(wid)
    return {"ok": True}


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

            # needs (per-species discovery) + per-person decay rates, so they drain over sim time
            yield sse({"type": "status", "message": "discovering needs…"})
            disc = needs.discover_needs(SERVER, species, n_extra=3)
            need_list = disc["have"] + disc["extra"]
            yield sse({"type": "needs", "needs": need_list})
            yield sse({"type": "status", "message": "timing needs…"})
            person_desc = f"{nm}, a {gender} {species}. {ctx['personality']}"
            rr = needs.discover_rates(SERVER, person_desc, need_list, samples=3)
            rates = {n: rr["rates"][n]["decay_per_hour"] for n in need_list}
            yield sse({"type": "rates", "rates": rates, "wake_hours": rr["wake_hours"]})
            yield sse({"type": "status", "message": "finding need thresholds…"})
            thresholds = _thresholds_for(need_list)        # per-need deadband (cached across characters)
            yield sse({"type": "thresholds", "thresholds": thresholds})

            cid = store.new_id(8)
            store.append(wid, {"type": "character", "cid": cid, "role": "location" if at else "seed",
                               "at": at, "world": setting, "species": species, "gender": gender,
                               "name": nm, "appearance": ctx["appearance"],
                               "personality": ctx["personality"], "backstory": ctx["backstory"],
                               "nocturnal": noc, "needs": need_list, "rates": rates,
                               "wake_hours": rr["wake_hours"], "thresholds": thresholds, "prompts": prompts})
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


# ------------------------------------------- per-location NPC counts over the day (gen_number)
@app.route("/api/world/<wid>/npccounts")
def api_npc_counts(wid):
    w = store.load_world(wid)
    loc = request.args.get("loc", "").strip()
    samples = int(request.args.get("samples", "10"))

    @stream_with_context
    def gen():
        if not w:
            yield sse({"type": "error", "message": "World not found."}); return
        if not loc:
            yield sse({"type": "error", "message": "No location."}); return
        try:
            q = wr.npc_count_prompt(w["prompt"], loc)         # distinct NPCs over a full day
            res = SERVER.gen_number_median(q, samples=samples)   # sample 10x, take the median
            if res:
                count, lo, hi = round(res["median"]), round(res["lo"]), round(res["hi"])
                store.append(wid, {"type": "npc_total", "location": loc, "count": count,
                                   "lo": lo, "hi": hi, "n": res["n"]})
                yield sse({"type": "npc", "location": loc, "count": count, "lo": lo, "hi": hi,
                           "n": res["n"], "samples": [s["raw"] for s in res["samples"]], "prompt": q})
            else:
                yield sse({"type": "npc", "location": loc, "count": None, "prompt": q})
            yield sse({"type": "done"})
        except Exception as e:
            yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return Response(gen(), mimetype="text/event-stream")


# ------------------------------------------------ per-location object types (re-sample + dedup)
@app.route("/api/world/<wid>/objects")
def api_objects(wid):
    w = store.load_world(wid)
    loc = request.args.get("loc", "").strip()
    n = int(request.args.get("n", "12"))

    @stream_with_context
    def gen():
        if not w:
            yield sse({"type": "error", "message": "World not found."}); return
        if not loc:
            yield sse({"type": "error", "message": "No location."}); return
        try:
            q = wr.objects_prompt(w["prompt"], loc)
            items = []
            for obj in wr.iter_objects(SERVER, w["prompt"], loc, n=n):
                items.append(obj)
                yield sse({"type": "object", "value": obj, "prompt": q})
            store.append(wid, {"type": "objects", "location": loc, "items": items, "prompt": q})
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
