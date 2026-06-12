"""
sim.py — the live colony sim behind the web UI. Placed characters become agents that walk (A*, via
pathfinding.py) to placed items to satisfy needs that drop below a deadband threshold, then refill
them — the colony.py utility loop, wired to stored world data (characters' needs/rates, items'
per-species affordances). State is in-memory per world id; the UI steps it and renders the result.

Time: 1 tick per game-minute, matching the UI clock. Needs decay by rate_per_hour/60 per tick. Movement
is WALK_CELLS cells/tick along the cached path. (Rates below are all PER-GAME-MINUTE now, so the colony
behaves identically to the old 5-ticks/min model — just clocked coarser, advancing the clock faster.)
"""
import pathfinding
import store

TICKS_PER_HOUR = 60
TICKS_PER_MIN = 1              # 1 tick per game-minute (1/GAME_MIN_PER_TICK) -> duration_min ticks
GAME_MIN_PER_TICK = 1.0
DEFAULT_THRESH = 0.35          # need deadband when a character has no baked per-need threshold
DEFAULT_DURATION_MIN = 5       # activity length for an item with no baked duration
WALK_CELLS = 15                # cells walked per tick (15 cells/game-min, = old 3 cells x 5 ticks/min)
START_GAME_MIN = 8 * 60
AMBIENT_FILL_PER_TICK = 0.25   # in-field refill RATE per tick = strength x this (0.25/game-min, balances decay)

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
    """Placed item instances as targets: [{pid, x, y, name, affords, durations, radii, consumable}]."""
    items = world.get("items", {})
    out = []
    for pid, ip in world.get("item_placements", {}).items():
        it = items.get(ip["iid"])
        if it:
            out.append({"pid": pid, "x": int(ip["x"]), "y": int(ip["y"]), "name": it["name"],
                        "affords": it.get("affords", {}), "durations": it.get("durations", {}),
                        "radii": it.get("radii", {}), "consumable": bool(it.get("consumable"))})
    return out


def _all_affords(item):
    """Union of needs an item serves across species -> {need: max refill/strength}. A physical field
    doesn't care which species stands in it, so ambient strength is the max over species."""
    a = item.get("affords", {}) or {}
    if a and all(isinstance(v, dict) for v in a.values()):     # per-species {sp: {need: r}}
        out = {}
        for sub in a.values():
            for n, r in sub.items():
                out[n] = max(out.get(n, 0.0), r)
        return out
    return dict(a)                                             # legacy flat {need: r}


def _providers(items, ambient_needs):
    """[{x, y, name, need, radius, strength}] for every placed item that PROVIDES an ambient need. A
    baked radius is REQUIRED — that's the mark of an item gated through the ambient provider-path, so a
    stale item with leftover use-gate affords (an old chair with 'shelter') emits no field."""
    out = []
    for it in items:
        radii = it.get("radii", {}) or {}
        for n, strength in _all_affords(it).items():
            if n in ambient_needs and strength > 0 and n in radii:
                out.append({"x": it["x"], "y": it["y"], "name": it["name"], "need": n,
                            "radius": radii[n], "strength": strength})
    return out


def _field(providers, n, x, y):
    """Strongest field of need `n` covering (x, y) — 0.0 if outside every provider's radius (cells)."""
    best = 0.0
    for p in providers:
        if p["need"] == n and (p["x"] - x) ** 2 + (p["y"] - y) ** 2 <= p["radius"] ** 2:
            best = max(best, p["strength"])
    return best


def _nearest_provider(providers, n, x, y):
    best, bestd = None, None
    for p in providers:
        if p["need"] == n:
            d = abs(p["x"] - x) + abs(p["y"] - y)
            if bestd is None or d < bestd:
                best, bestd = p, d
    return best


def _choose(agent, items, ambient_needs):
    """Best (item, need) for an ACTIVE need to act on: max over placed items AND the below-threshold
    non-ambient needs they serve of refill x urgency(1-level), distance-discounted. (None, None) if
    nothing. Ambient needs are excluded here — they're met by a field (see _decide), not by using an item."""
    best, best_score, best_need = None, 0.0, None
    for it in items:
        af = _affords_for(it, agent["species"])
        d = abs(agent["x"] - it["x"]) + abs(agent["y"] - it["y"])
        for n, r in af.items():
            if n in ambient_needs:
                continue
            if n in agent["needs"] and agent["needs"][n] < _thresh(agent, n):
                score = (r * (1.0 - agent["needs"][n])) / (1.0 + 0.01 * d)
                if score > best_score:
                    best, best_score, best_need = it, score, n
    return best, best_need


def _decide(agent, items, ambient_needs, providers):
    """Most worthwhile thing to do now, comparing two kinds on one urgency scale (refill/strength x
    (1-level), distance-discounted): an ACTIVE item-use (walk to it, use it for a duration, refill the
    one need) or an AMBIENT move-into-a-field (walk into the nearest provider's radius, then stand while
    the passive field refills the need — no fixed duration). Returns an action dict or None."""
    best_score, act = 0.0, None
    it, need = _choose(agent, items, ambient_needs)          # active candidate
    if it:
        af = _affords_for(it, agent["species"])
        d = abs(agent["x"] - it["x"]) + abs(agent["y"] - it["y"])
        best_score = (af.get(need, 0.0) * (1.0 - agent["needs"][need])) / (1.0 + 0.01 * d)
        act = {"kind": "active", "name": it["name"], "tx": it["x"], "ty": it["y"], "need": need,
               "refill": af.get(need, 0.0), "dur": (it.get("durations") or {}).get(need),
               "pid": it.get("pid"), "consumable": it.get("consumable", False)}
    for n in ambient_needs:                                  # ambient candidates: low + not already covered
        if (n in agent["needs"] and agent["needs"][n] < _thresh(agent, n)
                and _field(providers, n, agent["x"], agent["y"]) <= 0.0):
            p = _nearest_provider(providers, n, agent["x"], agent["y"])
            if p:
                d = abs(agent["x"] - p["x"]) + abs(agent["y"] - p["y"])
                score = (p["strength"] * (1.0 - agent["needs"][n])) / (1.0 + 0.01 * d)
                if score > best_score:
                    best_score = score
                    act = {"kind": "ambient", "name": p["name"], "tx": p["x"], "ty": p["y"], "need": n}
    return act


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
    ambient_needs = {n for n, m in world.get("need_modes", {}).items() if m.get("mode") == "ambient"}
    providers = _providers(items, ambient_needs)
    consumed = []
    for _ in range(max(1, ticks)):
        sim["game_min"] += GAME_MIN_PER_TICK
        for a in sim["agents"].values():
            serving = a["busy_affords"] if a["busy"] > 0 else {}
            for n in a["needs"]:                                  # decay (except what you're using)
                if n not in serving:
                    a["needs"][n] = max(0.0, a["needs"][n] - a["rates"].get(n, 0.0) / TICKS_PER_HOUR)
            for n in ambient_needs:                               # passive: standing in a field refills it
                if n in a["needs"]:                               # (positional — runs whatever you're doing)
                    f = _field(providers, n, a["x"], a["y"])
                    if f > 0.0:
                        a["needs"][n] = min(1.0, a["needs"][n] + f * AMBIENT_FILL_PER_TICK)
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
            if a["action"] is None:                               # decide: best active-use or ambient-field
                act = _decide(a, items, ambient_needs, providers)
                if act:
                    a["action"] = act
                    a["path"] = pathfinding.astar((a["x"], a["y"]), (act["tx"], act["ty"]))[1:]
            if a["action"]:                                       # walk toward the target
                act = a["action"]
                for _ in range(WALK_CELLS):
                    if a["path"]:
                        a["x"], a["y"] = a["path"].pop(0)
                if act["kind"] == "ambient":                      # arrive = inside the field; passive refill takes over
                    if _field(providers, act["need"], a["x"], a["y"]) > 0.0 or (a["x"], a["y"]) == (act["tx"], act["ty"]):
                        a["action"], a["path"] = None, []         # stand here (no busy duration — the field does it)
                    elif not a["path"]:
                        a["action"] = None                        # unreachable -> give up
                elif (a["x"], a["y"]) == (act["tx"], act["ty"]):  # active: begin the activity on arrival
                    dur = max(1, round((act["dur"] or DEFAULT_DURATION_MIN) * TICKS_PER_MIN))
                    a["busy"], a["busy_dur"] = dur, dur
                    a["busy_affords"] = {act["need"]: act["refill"]}   # refill ONLY the driving need
                    a["busy_name"] = act["name"]
                    a["busy_pid"], a["busy_consumable"] = act["pid"], act["consumable"]
                    a["action"], a["path"] = None, []
                elif not a["path"]:                               # unreachable -> give up
                    a["action"] = None
    st = _state(sim, providers, ambient_needs)
    st["consumed"] = consumed
    return st


def _doing(a, providers, ambient_needs):
    if a["busy"] > 0:
        return "using " + a["busy_name"]
    if a["action"]:
        return "→ " + a["action"]["name"]
    for n in ambient_needs:                                   # idle but standing in a field, still recovering
        if n in a["needs"] and a["needs"][n] < 1.0 and _field(providers, n, a["x"], a["y"]) > 0.0:
            return "↺ " + n
    return "idle"


def _state(sim, providers=(), ambient_needs=()):
    return {"game_min": sim["game_min"],
            "agents": {cid: {"x": a["x"], "y": a["y"],
                             "needs": {n: round(v, 4) for n, v in a["needs"].items()},
                             "doing": _doing(a, providers, ambient_needs)}
                       for cid, a in sim["agents"].items()}}


def reset(wid):
    _SIMS.pop(wid, None)
