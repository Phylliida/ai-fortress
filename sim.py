"""
sim.py — the live colony sim behind the web UI. Placed characters become agents that walk (A*, via
pathfinding.py) to placed items to satisfy needs that drop below a deadband threshold, then refill
them — the colony.py utility loop, wired to stored world data (characters' needs/rates, items'
per-species affordances). State is in-memory per world id; the UI steps it and renders the result.

Time: 5 ticks per game-minute (12 game-seconds/tick), matching colony.py / the UI clock. Needs decay
by rate_per_hour/300 per tick. Movement is WALK_CELLS cells/tick along the cached path.
"""
import pathfinding

TICKS_PER_HOUR = 300
GAME_MIN_PER_TICK = 0.2
DEFAULT_THRESH = 0.35          # need deadband until per-need baked thresholds are stored on characters
WALK_CELLS = 3                 # cells walked per tick
START_GAME_MIN = 8 * 60

_SIMS = {}                     # wid -> {"game_min": float, "agents": {cid: agent}}


def _new_agent(x, y, ch):
    return {"x": int(x), "y": int(y),
            "needs": {n: 1.0 for n in ch.get("needs", [])},
            "rates": ch.get("rates", {}) or {},
            "thresholds": ch.get("thresholds", {}) or {},
            "species": ch.get("species", "human"),
            "action": None, "path": []}


def _thresh(agent, need):
    return agent["thresholds"].get(need, DEFAULT_THRESH)


def _affords_for(item, species):
    a = item.get("affords", {}) or {}
    for v in a.values():
        return a.get(species, {}) if isinstance(v, dict) else a   # per-species vs legacy flat
    return {}


def _items(world):
    """Placed item instances as targets: [{x, y, name, affords}]."""
    items = world.get("items", {})
    out = []
    for ip in world.get("item_placements", {}).values():
        it = items.get(ip["iid"])
        if it:
            out.append({"x": int(ip["x"]), "y": int(ip["y"]), "name": it["name"], "affords": it.get("affords", {})})
    return out


def _choose(agent, items):
    """Highest-utility placed item: sum of refill x urgency(1-level) over the needs it serves that are
    BELOW threshold (deadband), distance-discounted. None if nothing is worth doing."""
    best, best_score = None, 0.0
    for it in items:
        af = _affords_for(it, agent["species"])
        val = sum(r * (1.0 - agent["needs"].get(n, 1.0))
                  for n, r in af.items()
                  if n in agent["needs"] and agent["needs"][n] < _thresh(agent, n))
        if val <= 0:
            continue
        d = abs(agent["x"] - it["x"]) + abs(agent["y"] - it["y"])
        score = val / (1.0 + 0.01 * d)
        if score > best_score:
            best, best_score = it, score
    return best


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
    for _ in range(max(1, ticks)):
        sim["game_min"] += GAME_MIN_PER_TICK
        for a in sim["agents"].values():
            for n in a["needs"]:                                  # decay
                a["needs"][n] = max(0.0, a["needs"][n] - a["rates"].get(n, 0.0) / TICKS_PER_HOUR)
            if a["action"] is None:                               # decide
                tgt = _choose(a, items)
                if tgt:
                    a["action"] = {"name": tgt["name"], "tx": tgt["x"], "ty": tgt["y"],
                                   "affords": _affords_for(tgt, a["species"])}
                    a["path"] = pathfinding.astar((a["x"], a["y"]), (tgt["x"], tgt["y"]))[1:]
            if a["action"]:                                       # walk, then use on arrival
                for _ in range(WALK_CELLS):
                    if a["path"]:
                        a["x"], a["y"] = a["path"].pop(0)
                if (a["x"], a["y"]) == (a["action"]["tx"], a["action"]["ty"]):
                    for n, r in a["action"]["affords"].items():
                        if n in a["needs"]:
                            a["needs"][n] = min(1.0, a["needs"][n] + r)
                    a["action"], a["path"] = None, []
                elif not a["path"]:                               # unreachable -> give up
                    a["action"] = None
    return _state(sim)


def _state(sim):
    return {"game_min": sim["game_min"],
            "agents": {cid: {"x": a["x"], "y": a["y"],
                             "needs": {n: round(v, 4) for n, v in a["needs"].items()},
                             "doing": ("→ " + a["action"]["name"]) if a["action"] else "idle"}
                       for cid, a in sim["agents"].items()}}


def reset(wid):
    _SIMS.pop(wid, None)
