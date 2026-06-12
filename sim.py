"""
sim.py — the live colony sim behind the web UI. Placed characters become agents that walk (A*, via
pathfinding.py) to placed items to satisfy needs that drop below a deadband threshold, then refill
them — the colony.py utility loop, wired to stored world data (characters' needs/rates, items'
per-species affordances). State is in-memory per world id; the UI steps it and renders the result.

Time: 5 ticks per game-minute (12 game-seconds/tick), matching colony.py / the UI clock. Needs decay
by rate_per_hour/300 per tick. Movement is WALK_CELLS cells/tick along the cached path.
"""
import pathfinding
import store

TICKS_PER_HOUR = 300
TICKS_PER_MIN = 5              # 5 ticks per game-minute (1/GAME_MIN_PER_TICK) -> duration_min * 5 = ticks
GAME_MIN_PER_TICK = 0.2
DEFAULT_THRESH = 0.35          # need deadband when a character has no baked per-need threshold
DEFAULT_DURATION_MIN = 5       # activity length for an item with no baked duration
WALK_CELLS = 3                 # cells walked per tick
START_GAME_MIN = 8 * 60

_SIMS = {}                     # wid -> {"game_min": float, "agents": {cid: agent}}


def _new_agent(x, y, ch):
    return {"x": int(x), "y": int(y),
            "needs": {n: 1.0 for n in ch.get("needs", [])},
            "rates": ch.get("rates", {}) or {},
            "thresholds": ch.get("thresholds", {}) or {},
            "species": ch.get("species", "human"),
            "action": None, "path": [],
            "busy": 0, "busy_dur": 1, "busy_affords": {}, "busy_name": "",
            "busy_pid": None, "busy_consumable": False}


def _thresh(agent, need):
    return agent["thresholds"].get(need, DEFAULT_THRESH)


def _affords_for(item, species):
    a = item.get("affords", {}) or {}
    for v in a.values():
        return a.get(species, {}) if isinstance(v, dict) else a   # per-species vs legacy flat
    return {}


def _items(world):
    """Placed item instances as targets: [{pid, x, y, name, affords, duration_min, consumable}]."""
    items = world.get("items", {})
    out = []
    for pid, ip in world.get("item_placements", {}).items():
        it = items.get(ip["iid"])
        if it:
            out.append({"pid": pid, "x": int(ip["x"]), "y": int(ip["y"]), "name": it["name"],
                        "affords": it.get("affords", {}), "durations": it.get("durations", {}),
                        "consumable": bool(it.get("consumable"))})
    return out


def _choose(agent, items):
    """Best (item, need) to act on: max over placed items AND the below-threshold needs they serve of
    refill x urgency(1-level), distance-discounted. Returns (item, need) — the agent uses the item FOR
    that one need (its duration, its refill), so a multi-need item doesn't borrow another need's time
    (a royal bed for novelty != an 8h sleep). (None, None) if nothing is worth doing."""
    best, best_score, best_need = None, 0.0, None
    for it in items:
        af = _affords_for(it, agent["species"])
        d = abs(agent["x"] - it["x"]) + abs(agent["y"] - it["y"])
        for n, r in af.items():
            if n in agent["needs"] and agent["needs"][n] < _thresh(agent, n):
                score = (r * (1.0 - agent["needs"][n])) / (1.0 + 0.01 * d)
                if score > best_score:
                    best, best_score, best_need = it, score, n
    return best, best_need


def _reconcile(sim, world):
    """Add agents for newly-placed characters, drop agents whose character left the map."""
    placed = world.get("placements", {})
    bycid = {c["cid"]: c for c in world.get("characters", [])}
    for cid, pos in placed.items():
        if cid not in sim["agents"] and cid in bycid:
            sim["agents"][cid] = _new_agent(pos["x"], pos["y"], bycid[cid])
    for cid in list(sim["agents"]):
        if cid not in placed:
            del sim["agents"][cid]


def step_world(world, ticks=1):
    """Advance the world's sim `ticks` ticks and return its state."""
    sim = _SIMS.setdefault(world["id"], {"game_min": float(START_GAME_MIN), "agents": {}})
    _reconcile(sim, world)
    items = _items(world)
    consumed = []
    for _ in range(max(1, ticks)):
        sim["game_min"] += GAME_MIN_PER_TICK
        for a in sim["agents"].values():
            serving = a["busy_affords"] if a["busy"] > 0 else {}
            for n in a["needs"]:                                  # decay (except what you're using)
                if n not in serving:
                    a["needs"][n] = max(0.0, a["needs"][n] - a["rates"].get(n, 0.0) / TICKS_PER_HOUR)
            if a["busy"] > 0:                                     # mid-activity: spread refill over its span
                for n, r in serving.items():
                    if n in a["needs"]:
                        a["needs"][n] = min(1.0, a["needs"][n] + r / a["busy_dur"])
                a["busy"] -= 1
                if a["busy"] == 0:
                    if a["busy_consumable"] and a["busy_pid"]:     # used up -> remove the instance (persist)
                        store.append(world["id"], {"type": "item_place", "pid": a["busy_pid"], "x": None, "y": None})
                        consumed.append(a["busy_pid"])
                        items[:] = [it for it in items if it["pid"] != a["busy_pid"]]
                    a["busy_affords"], a["busy_name"] = {}, ""
                    a["busy_pid"], a["busy_consumable"] = None, False
                continue                                          # committed to the full duration
            if a["action"] is None:                               # decide: best (item, need)
                tgt, need = _choose(a, items)
                if tgt:
                    af = _affords_for(tgt, a["species"])
                    a["action"] = {"name": tgt["name"], "tx": tgt["x"], "ty": tgt["y"],
                                   "need": need, "refill": af.get(need, 0.0),
                                   "dur": (tgt.get("durations") or {}).get(need),
                                   "pid": tgt.get("pid"), "consumable": tgt.get("consumable", False)}
                    a["path"] = pathfinding.astar((a["x"], a["y"]), (tgt["x"], tgt["y"]))[1:]
            if a["action"]:                                       # walk; begin the activity on arrival
                for _ in range(WALK_CELLS):
                    if a["path"]:
                        a["x"], a["y"] = a["path"].pop(0)
                if (a["x"], a["y"]) == (a["action"]["tx"], a["action"]["ty"]):
                    dur = max(1, round((a["action"]["dur"] or DEFAULT_DURATION_MIN) * TICKS_PER_MIN))
                    a["busy"], a["busy_dur"] = dur, dur
                    a["busy_affords"] = {a["action"]["need"]: a["action"]["refill"]}   # refill ONLY the driving need
                    a["busy_name"] = a["action"]["name"]
                    a["busy_pid"], a["busy_consumable"] = a["action"]["pid"], a["action"]["consumable"]
                    a["action"], a["path"] = None, []
                elif not a["path"]:                               # unreachable -> give up
                    a["action"] = None
    st = _state(sim)
    st["consumed"] = consumed
    return st


def _doing(a):
    if a["busy"] > 0:
        return "using " + a["busy_name"]
    if a["action"]:
        return "→ " + a["action"]["name"]
    return "idle"


def _state(sim):
    return {"game_min": sim["game_min"],
            "agents": {cid: {"x": a["x"], "y": a["y"],
                             "needs": {n: round(v, 4) for n, v in a["needs"].items()},
                             "doing": _doing(a)}
                       for cid, a in sim["agents"].items()}}


def reset(wid):
    _SIMS.pop(wid, None)
