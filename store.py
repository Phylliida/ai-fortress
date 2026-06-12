"""
Append-only per-world storage. Each world is worlds/<id>.jsonl — one typed JSON record per
line, so every write is a single appended line (no rewrite). The world's current state is
reconstructed by replaying the file.

Record types:
  {"type":"world",     "id", "prompt", "created"}
  {"type":"character",  "cid", "role":"seed"|"location", "at":<loc|null>, "world",
                        "species", "gender", "name", "appearance", "personality", "backstory"}
  {"type":"locations",  "kept":[ ...top-5... ]}
  {"type":"place",      "cid", "x", "y"}   # position on the 500x500 world map; last write wins
                                            # (x or y null = removed from the map)
  {"type":"item",       "iid", "name", "species", "affords":{need:refill},  # an item TEMPLATE (palette)
                        "durations":{need:min}, "durations_ok":{need:conf},   # per-need activity length
                        "radii":{need:cells}, "consumable"}   # ambient-need field radius (provider)
  {"type":"item_duration", "iid", "need", "duration_min"}   # per-need manual duration override
  {"type":"need_mode",  "need", "mode", "conf", "manual"}   # how a need is met (consume/restore/
                                            # ambient/social/experiential); last write wins per need
  {"type":"item_delete", "iid"}   # delete an item TEMPLATE from the palette (+ its placed instances)
  {"type":"item_place", "pid", "iid", "x", "y"}   # one instance on the map (many per template;
                                                   # last write wins per pid; x or y null = removed)
  {"type":"query",      "t", "prompt", "n_predict", "grammar", "content", "top"}  # base-model query
                                                   # log (audit/debug); ignored by load_world, read via
                                                   # get_log()
"""
import os
import json
import uuid
import datetime

WORLDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlds")
os.makedirs(WORLDS_DIR, exist_ok=True)


def _path(world_id):
    return os.path.join(WORLDS_DIR, world_id + ".jsonl")


def new_id(n=12):
    return uuid.uuid4().hex[:n]


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_world(prompt):
    world_id = new_id()
    rec = {"type": "world", "id": world_id, "prompt": prompt, "created": _now()}
    with open(_path(world_id), "w") as f:
        f.write(json.dumps(rec) + "\n")
    return world_id


def append(world_id, record):
    with open(_path(world_id), "a") as f:
        f.write(json.dumps(record) + "\n")


def _records(world_id):
    p = _path(world_id)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_world(world_id):
    """Replay the file into the world's current state (None if it doesn't exist)."""
    recs = _records(world_id)
    if not recs:
        return None
    world = {"id": world_id, "prompt": None, "created": None,
             "characters": [], "locations": [], "npc_totals": {}, "objects": {}, "placements": {},
             "items": {}, "item_placements": {}, "need_modes": {}}
    for r in recs:
        t = r.get("type")
        if t == "world":
            world["prompt"], world["created"] = r["prompt"], r.get("created")
        elif t == "character":
            world["characters"].append(r)
        elif t == "locations":
            world["locations"] = r["kept"]
        elif t == "place":
            if r.get("x") is None or r.get("y") is None:
                world["placements"].pop(r["cid"], None)        # null coords = taken off the map
            else:
                world["placements"][r["cid"]] = {"x": r["x"], "y": r["y"]}
        elif t == "item":
            world["items"][r["iid"]] = {"name": r["name"], "species": r.get("species"),
                                        "affords": r.get("affords", {}),
                                        "durations": r.get("durations", {}),       # {need: minutes} (active)
                                        "durations_ok": r.get("durations_ok", {}),  # {need: gate conf}
                                        "radii": r.get("radii", {}),                # {need: cells} (ambient)
                                        "consumable": bool(r.get("consumable"))}
        elif t == "item_duration":
            if r.get("need") is not None and r["iid"] in world["items"]:   # skip legacy per-item records
                world["items"][r["iid"]].setdefault("durations", {})[r["need"]] = r["duration_min"]
                world["items"][r["iid"]].setdefault("durations_ok", {})[r["need"]] = 1.0  # set -> trusted
        elif t == "need_mode":
            world["need_modes"][r["need"]] = {"mode": r["mode"], "conf": r.get("conf"),
                                              "manual": bool(r.get("manual"))}
        elif t == "item_delete":
            world["items"].pop(r["iid"], None)
            for pid in [p for p, ip in world["item_placements"].items() if ip.get("iid") == r["iid"]]:
                world["item_placements"].pop(pid, None)        # drop its placed instances too
        elif t == "item_place":
            if r.get("x") is None or r.get("y") is None:
                world["item_placements"].pop(r["pid"], None)
            else:
                world["item_placements"][r["pid"]] = {"iid": r["iid"], "x": r["x"], "y": r["y"]}
        elif t == "npc_total":
            world["npc_totals"][r["location"]] = {"count": r["count"], "lo": r.get("lo"), "hi": r.get("hi")}
        elif t == "objects":
            world["objects"][r["location"]] = {"items": r["items"], "prompt": r.get("prompt")}
    world["species"] = sorted({c.get("species", "human") for c in world["characters"]})
    return world


def get_log(world_id, since=0, cap=500):
    """Base-model query records for a world, each tagged with a stable 1-based index `i`. since>0
    returns only records after that index (polling); since<=0 returns the last `cap`."""
    p = _path(world_id)
    if not os.path.exists(p):
        return []
    out, i = [], 0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") != "query":
                continue
            i += 1
            if i > since:
                r["i"] = i
                out.append(r)
    return out[-cap:] if since <= 0 else out


def get_character(world_id, cid):
    for r in _records(world_id):
        if r.get("type") == "character" and r.get("cid") == cid:
            return r
    return None


def list_worlds():
    """Menu: one entry per world file (id, prompt, created, #characters, has_locations)."""
    out = []
    for fn in os.listdir(WORLDS_DIR):
        if not fn.endswith(".jsonl"):
            continue
        recs = _records(fn[:-6])
        if not recs:
            continue
        w0 = recs[0]
        out.append({
            "id": fn[:-6], "prompt": w0.get("prompt", ""), "created": w0.get("created"),
            "n_characters": sum(1 for r in recs if r.get("type") == "character"),
            "has_locations": any(r.get("type") == "locations" for r in recs),
        })
    out.sort(key=lambda w: w.get("created") or "", reverse=True)
    return out
