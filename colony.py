"""
colony.py — the headless colony-sim tick (the RUN-TIME half; the LLM-as-designer authored the
constants over in needs.py). Agents live on a grid with need bars that DECAY each tick; an agent
picks the reachable object that best relieves its most-depleted needs (Sims-style utility: the
object whose refills, weighted by how depleted each need is, score highest), walks one cell toward
it, and then *performs the activity over time*. Pure arithmetic — NO LLM in the loop.

TIME: one tick = 12 seconds (5 ticks/minute, 300/hour). Need decay rates are PER HOUR — exactly
what needs.discover_rates produces — and step() scales them by dt = hours-per-tick, so a ~3-minute
walk barely dents any need.

ACTIVITIES TAKE TIME: using an object is NOT instantaneous. Each object has a duration (eat ~20 min,
sleep ~8 h, socialize ~45 min); the refill is spread gradually across that span, and the need being
serviced does NOT decay while you're at it (you don't get hungrier while eating). This is what stops
frantic topping-up — committing to a real-time activity has a real-time cost — without needing any
urgency curve. Durations are bake-able too (needs.bake_durations / gen_number "how many minutes...").

Inputs are exactly what the bake produces:
  - need levels + per-HOUR decay        <- needs.discover_rates  (decay_per_hour)
  - object affordances {need: refill}    <- needs.bake_affordances
  - per-object activity duration (min)   <- needs.bake_durations

Need level L in [0,1]: 1 = fully satisfied, 0 = critical. urgency(need) = 1 - L.
Run `python3 colony.py` for a one-agent demo day.
"""

TICKS_PER_MIN = 5
TICKS_PER_HOUR = TICKS_PER_MIN * 60          # 300 ticks/hour; one tick = 12 s
HOURS_PER_TICK = 1.0 / TICKS_PER_HOUR


class Agent:
    def __init__(self, name, species, x, y, needs, decay):
        self.name, self.species = name, species
        self.x, self.y = x, y
        self.needs = dict(needs)      # {need: level 0..1}
        self.decay = dict(decay)      # {need: drop per HOUR} (step scales by dt)
        self.action = None            # {"kind", "target": (x,y), "obj": Obj} | None — current goal
        self.busy = 0                 # ticks left in the in-progress activity
        self.busy_obj = None          # the Obj being used right now
        self.just_started = None      # kind name on the tick an activity begins (for display)
        self.log = "spawned"


class Obj:
    def __init__(self, kind, x, y, affords, duration_min):
        self.kind, self.x, self.y = kind, x, y
        self.affords = dict(affords)  # {need: refill 0..1} — total restored over the whole activity
        self.duration_min = duration_min  # how long the activity takes (bake-able)


class Colony:
    def __init__(self, w, h, agents, objects, dt=HOURS_PER_TICK):
        self.w, self.h = w, h
        self.agents, self.objects = agents, objects
        self.dt, self.tick = dt, 0    # dt = sim-HOURS per tick


def _dist(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


LOW_WATER = 0.35   # a need only "demands" attention once it drops below this; above it the agent
#                    leaves it alone. Deadband on the NEED LEVEL — gates WHEN she acts, independent
#                    of how big an object's refill is, so a high-refill bed can't lure her into a
#                    full night's sleep when she's only 10% tired. (Not an urgency curve — a
#                    thermostat: act below low-water, stop when full.)


def object_value(agent, obj):
    """Sims advertise: sum over the needs this object serves of refill x urgency (1 - level), but
    ONLY needs that have dropped below LOW_WATER count — a comfortable need advertises nothing."""
    return sum(r * (1.0 - agent.needs.get(n, 1.0))
               for n, r in obj.affords.items()
               if agent.needs.get(n, 1.0) < LOW_WATER)


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
    """One tick: needs decay (except the one being serviced), then either continue an in-progress
    activity (gradual refill), walk toward a chosen object, or start the activity on arrival."""
    colony.tick += 1
    for a in colony.agents:
        a.just_started = None
        serving = set(a.busy_obj.affords) if (a.busy > 0 and a.busy_obj) else set()
        for n in a.needs:                               # decay everything you're NOT currently doing
            if n not in serving:
                a.needs[n] = max(0.0, a.needs[n] - a.decay.get(n, 0.0) * colony.dt)

        if a.busy > 0:                                  # mid-activity: spread the refill over its span
            o = a.busy_obj
            dur = max(1, round(o.duration_min * TICKS_PER_MIN))
            for n, r in o.affords.items():
                if n in a.needs:
                    a.needs[n] = min(1.0, a.needs[n] + r / dur)
            a.busy -= 1
            a.log = f"using {o.kind}"
            if a.busy == 0:                             # committed to the FULL duration — no bailing
                a.busy_obj, a.action = None, None       # early (that's what stops constant napping)
            continue

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
        if (a.x, a.y) == (tx, ty):                       # arrived -> BEGIN the activity (takes time)
            o = a.action["obj"]
            a.busy = max(1, round(o.duration_min * TICKS_PER_MIN))
            a.busy_obj, a.just_started, a.log = o, o.kind, f"using {o.kind}"
        else:
            a.log = f"walking to {a.action['kind']}"


if __name__ == "__main__":
    from collections import Counter

    def clock(tick):
        m = tick / TICKS_PER_MIN
        return f"{int(m // 60) % 24:02d}:{int(m % 60):02d}"

    # A baked profile of the kind needs.discover_rates / bake_affordances / bake_durations produce —
    # hand-set so the demo is fast + deterministic. Decay PER HOUR; durations in MINUTES.
    maria = Agent("Maria", "human", 4, 4,
                  needs={"food": 0.9, "sleep": 0.95, "social": 0.7},
                  decay={"food": 0.12, "sleep": 0.06, "social": 0.20})
    objects = [
        Obj("hearth", 0, 0, {"food": 0.94},   duration_min=20),   # a meal: 20 min
        Obj("bed",    7, 7, {"sleep": 0.89},  duration_min=480),  # a night: 8 h
        Obj("tavern", 0, 7, {"social": 0.79}, duration_min=45),   # a visit: 45 min
    ]
    colony = Colony(8, 8, [maria], objects)             # dt defaults to 12 s/tick
    DAY = 24
    print(f"time    pos      food sleep social  | started this hour      | now")
    acts = Counter()
    for tick in range(1, DAY * TICKS_PER_HOUR + 1):
        step(colony)
        if maria.just_started:
            acts[maria.just_started] += 1
        if tick % TICKS_PER_HOUR == 0:
            n = maria.needs
            did = ", ".join(f"{k}×{c}" for k, c in acts.items()) or "—"
            print(f"{clock(tick)}  {str((maria.x, maria.y)):>7}  "
                  f"{n['food']:.2f} {n['sleep']:.2f} {n['social']:.2f}  | {did:21s} | {maria.log}")
            acts.clear()
