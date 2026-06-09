"""
needs.py — colony-sim need discovery (hybrid candidate model).

A fixed UNIVERSAL_CORE of single-word biological/psychological needs is yes/no-checked against
each SPECIES every time (a robot may not 'sleep', a fungus may not need a 'bathroom'), then
species-specific extra needs are generated and embed-deduped against the applying core. This
step finds WHICH needs a species has; rates/decay (gen_number) and object affordances
(gen_percent gated by a yes/no "applies?") come next. Built on baseModelPrimitives.

Needs are baked per SPECIES (shared by all members); how fast each decays is per-person (later).
"""
from baseModelPrimitives import iter_unique, LOCATION_GRAMMAR

# Fixed universal core: single-word needs swept (yes/no) against every species, every time.
UNIVERSAL_CORE = [
    "food", "water", "sleep", "air", "warmth", "bathroom", "hygiene", "health",
    "safety", "shelter", "social", "love", "respect", "purpose", "novelty", "movement",
]


def need_applies_prompt(species, need):
    # "if it were real" sidesteps the model refusing needs to fictional beings (it fixated on
    # "dragons aren't real"); the counterfactual is harmless for species that already are real.
    return (f"If a {species} were real, would it have a {need} need "
            f"(something it must regularly satisfy)? Answer yes or no.\nAnswer:")


def extra_needs_prompt(species, have):
    return (f"A {species} has needs such as: {', '.join(have)}.\n"
            f"List other needs a {species} has, each a single word:\n"
            f"Another single-word need a {species} has is:\n-")


def discover_core(server, species, threshold=0.5):
    """yes/no-sweep the whole UNIVERSAL_CORE against `species`; each entry keeps P(yes)."""
    out = []
    for need in UNIVERSAL_CORE:
        p = server.yes_no_prob(need_applies_prompt(species, need))
        out.append({"need": need, "applies": p >= threshold, "p": round(p, 3)})
    return out


def discover_needs(server, species, n_extra=6, threshold=0.5):
    """Full hybrid discovery: core sweep + dedup-suggested single-word species-specific extras."""
    core = discover_core(server, species, threshold)
    have = [c["need"] for c in core if c["applies"]]
    # extras dedup against the FULL core (not just the applying ones) so a dropped-but-borderline
    # core word can't resurface as an "extra"; reject non-single-word / non-alphabetic junk.
    extra = [e.lower() for e in iter_unique(
        server, extra_needs_prompt(species, have), n=n_extra, grammar=LOCATION_GRAMMAR,
        max_len=20, seed=UNIVERSAL_CORE,
        reject=lambda x: len(x.split()) != 1 or not x.replace("-", "").replace("'", "").isalpha())]
    return {"species": species, "core": core, "have": have, "extra": extra}


# --- rates: PER-PERSON, since some people need more of a thing than others (a gregarious person
#     socializes more). Which needs a species has is per-species (above); how fast each drains is
#     per-person. Linear decay: N satisfactions across W waking hours => the bar empties every W/N
#     hours, i.e. decays N/W of full per hour.

def rate_prompt(person, need):
    """`person` is a description string — the richer it is, the more individuated the rate."""
    return (f"{person}\n"
            f"On a typical day, how many times does this person need to satisfy their {need} need?\n"
            f"Answer:")


def wake_hours_prompt(person):
    return (f"{person}\n"
            f"On a typical day, how many hours is this person awake (not asleep)?\nAnswer:")


def discover_rates(server, person, need_list, wake_hours=None, samples=5):
    """Per-person rates for each need in `need_list`. times/day via gen_number_median (robust),
    then linear decay over wake_hours/N hours. `wake_hours` is asked (gen_number) if not given.
    Returns {wake_hours, rates: {need: {per_day, hours_to_empty, decay_per_hour}}}."""
    w = wake_hours
    if w is None:
        r = server.gen_number_median(wake_hours_prompt(person), samples=samples)
        w = min(max(round(r["median"]), 1), 24) if r else 16
    rates = {}
    for need in need_list:
        r = server.gen_number_median(rate_prompt(person, need), samples=samples)
        n = max(r["median"], 0.1) if r else 1.0           # guard against 0 (no div-by-zero)
        rates[need] = {"per_day": round(n, 2),
                       "hours_to_empty": round(w / n, 2),
                       "decay_per_hour": round(n / w, 4)}
    return {"wake_hours": w, "rates": rates}


# --- affordances: how much an OBJECT-KIND refills a NEED (the Sims "advertise" values), baked per
#     (species, object-kind, need). Two steps: a yes/no GATE ("does it affect this need at all?")
#     filters irrelevant pairs (painting -> hunger), so gen_percent only runs on real affordances
#     (and never sees the fat-tail leak it has on irrelevant inputs).

def affordance_applies_prompt(species, obj_kind, need):
    # "relevant to satisfying" separates real affordances from irrelevant pairs far better than
    # "affect ... at all" (which biased to no — gated meal->food at 0.32); calibrated 9/9.
    return (f"For a {species}, is a {obj_kind} relevant to satisfying their {need} need? "
            f"Answer yes or no.\nAnswer:")


def affordance_amount_prompt(species, obj_kind, need):
    """Framed so a degree phrase is the natural continuation (for gen_percent)."""
    return (f"Q: How much does using a {obj_kind} satisfy a {species}'s {need} need?\n"
            f"A: It satisfies their {need} need")


def bake_affordance(server, species, obj_kind, need, gate_threshold=0.5):
    """One (species, obj_kind, need): gate (yes_no) then degree (gen_percent). Returns
    {applies, p_applies, refill} — refill 0.0 when the gate fails (no gen_percent call)."""
    p = server.yes_no_prob(affordance_applies_prompt(species, obj_kind, need))
    if p < gate_threshold:
        return {"applies": False, "p_applies": round(p, 3), "refill": 0.0}
    pct = server.gen_percent(affordance_amount_prompt(species, obj_kind, need))
    return {"applies": True, "p_applies": round(p, 3), "refill": round(pct["value"], 3) if pct else 0.0}


def bake_affordances(server, species, obj_kinds, need_list, gate_threshold=0.5):
    """Full advertise table for a species: {obj_kind: {need: refill}} keeping only pairs that pass
    the gate with refill > 0. This is exactly what an agent reads to choose its next action."""
    table = {}
    for kind in obj_kinds:
        served = {}
        for need in need_list:
            r = bake_affordance(server, species, kind, need, gate_threshold)
            if r["applies"] and r["refill"] > 0:
                served[need] = r["refill"]
        table[kind] = served
    return table
