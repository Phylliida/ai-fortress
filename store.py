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
                        "duration_min", "duration_ok"}  # activity length + sanity-gate confidence
  {"type":"item_duration", "iid", "duration_min"}   # manual duration override (trusted)
  {"type":"item_place", "pid", "iid", "x", "y"}   # one instance on the map (many per template;
                                                   # last write wins per pid; x or y null = removed)
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
             "items": {}, "item_placements": {}}
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
                                        "duration_min": r.get("duration_min"),
                                        "duration_ok": r.get("duration_ok")}
        elif t == "item_duration":
            if r["iid"] in world["items"]:
                world["items"][r["iid"]]["duration_min"] = r["duration_min"]
                world["items"][r["iid"]]["duration_ok"] = 1.0        # manually set -> trusted
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
