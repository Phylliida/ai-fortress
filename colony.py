"""
colony.py — the headless colony-sim tick (the RUN-TIME half; the LLM-as-designer authored the
constants over in needs.py). Agents live on a grid with need bars that DECAY each tick; an agent
picks the reachable object that best relieves its most-depleted needs (Sims-style utility: the
object whose refills, weighted by how depleted each need is, score highest), walks one cell toward
it, and uses it on arrival. Pure arithmetic — NO LLM in the loop.

Inputs are exactly what the bake produces:
  - need levels + per-tick decay   <- needs.discover_rates  (decay_per_hour)
  - object affordances {need: refill} <- needs.bake_affordances

Need level L in [0,1]: 1 = fully satisfied, 0 = critical. urgency(need) = 1 - L.
Run `python3 colony.py` for a one-agent demo day.
"""


class Agent:
    def __init__(self, name, species, x, y, needs, decay):
        self.name, self.species = name, species
        self.x, self.y = x, y
        self.needs = dict(needs)      # {need: level 0..1}
        self.decay = dict(decay)      # {need: drop per tick}
        self.action = None            # {"kind", "target": (x,y), "obj": Obj} | None
        self.log = "spawned"


class Obj:
    def __init__(self, kind, x, y, affords):
        self.kind, self.x, self.y = kind, x, y
        self.affords = dict(affords)  # {need: refill 0..1}


class Colony:
    def __init__(self, w, h, agents, objects, dt=1.0):
        self.w, self.h = w, h
        self.agents, self.objects = agents, objects
        self.dt, self.tick = dt, 0


def _dist(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def object_value(agent, obj):
    """Sims advertise: sum over the needs this object serves of refill x urgency (1 - level).
    A near-full need contributes ~nothing; a near-empty one it can refill scores high."""
    return sum(r * (1.0 - agent.needs.get(n, 1.0)) for n, r in obj.affords.items())


def choose_action(agent, colony, threshold=0.05, dist_discount=0.04):
    """Pick the highest-value object (mild distance discount), or None to idle if nothing's worth it."""
    best, best_score = None, threshold
    for o in colony.objects:
        v = object_value(agent, o)
        if v <= 0:
            continue
        score = v / (1.0 + dist_discount * _dist(agent.x, agent.y, o.x, o.y))
        if score > best_score:
            best, best_score = o, score
    return None if best is None else {"kind": best.kind, "target": (best.x, best.y), "obj": best}


def step(colony):
    """One tick: decay every need, (re)pick an action for idle agents, move one cell, use on arrival."""
    colony.tick += 1
    for a in colony.agents:
        for n in a.needs:
            a.needs[n] = max(0.0, a.needs[n] - a.decay.get(n, 0.0) * colony.dt)
        if a.action is None:
            a.action = choose_action(a, colony)
        if a.action is None:
            a.log = "idle (all needs ok)"
            continue
        tx, ty = a.action["target"]
        if a.x != tx:
            a.x += 1 if tx > a.x else -1
        elif a.y != ty:
            a.y += 1 if ty > a.y else -1
        if (a.x, a.y) == (tx, ty):                       # arrived -> use the object
            for n, r in a.action["obj"].affords.items():
                if n in a.needs:
                    a.needs[n] = min(1.0, a.needs[n] + r)
            a.log = f"used {a.action['kind']}"
            a.action = None                              # idle next tick -> re-choose
        else:
            a.log = f"walking to {a.action['kind']}"


if __name__ == "__main__":
    # A baked profile of the kind needs.discover_rates / bake_affordances produce — hand-set here
    # so the tick demo is fast + deterministic. (Maria-ish: social drains fastest, then food, sleep slow.)
    maria = Agent("Maria", "human", 4, 4,
                  needs={"food": 0.9, "sleep": 0.95, "social": 0.7},
                  decay={"food": 0.12, "sleep": 0.06, "social": 0.20})
    objects = [
        Obj("hearth", 0, 0, {"food": 0.94}),
        Obj("bed",    7, 7, {"sleep": 0.89}),
        Obj("tavern", 0, 7, {"social": 0.79}),
    ]
    colony = Colony(8, 8, [maria], objects, dt=1.0)
    print(f"{'h':>2} {'pos':>7}  food sleep social  | what Maria does")
    for _ in range(30):
        step(colony)
        n = maria.needs
        flag = "  <-- " + maria.log if maria.log.startswith("used") else ""
        print(f"{colony.tick:>2} {str((maria.x, maria.y)):>7}  "
              f"{n['food']:.2f}  {n['sleep']:.2f}  {n['social']:.2f}  | {maria.log}{flag}")
