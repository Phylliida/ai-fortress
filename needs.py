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

# Manual exclude list: needs dropped from discovery results for now. 'air' (+ gas synonyms) is too
# frequent an action to model usefully yet — gasses get their own treatment later. The core sweep
# still RECORDS these (for transparency); they're just filtered out of the actionable have/extra.
EXCLUDE_NEEDS = {"air", "oxygen", "breathing"}


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
    """Full hybrid discovery: core sweep + dedup-suggested single-word species-specific extras.
    EXCLUDE_NEEDS are dropped from the returned have/extra (the prompt still sees them so the model
    doesn't re-suggest them; the core sweep still records them)."""
    core = discover_core(server, species, threshold)
    have_all = [c["need"] for c in core if c["applies"]]
    have = [n for n in have_all if n not in EXCLUDE_NEEDS]
    # extras dedup against the FULL core (not just the applying ones) so a dropped-but-borderline
    # core word can't resurface as an "extra"; reject non-single-word / non-alphabetic junk / excludes.
    extra = [e.lower() for e in iter_unique(
        server, extra_needs_prompt(species, have_all), n=n_extra, grammar=LOCATION_GRAMMAR,
        max_len=20, seed=UNIVERSAL_CORE,
        reject=lambda x: len(x.split()) != 1 or not x.replace("-", "").replace("'", "").isalpha()
                         or x.lower() in EXCLUDE_NEEDS)]
    return {"species": species, "core": core, "have": have, "extra": extra}


# --- rates: PER-PERSON, since some people need more of a thing than others (a gregarious person
#     socializes more). Which needs a species has is per-species (above); how fast each drains is
#     per-person. Linear decay: N satisfactions across W waking hours => the bar empties every W/N
#     hours, i.e. decays N/W of full per hour.

# Few-shot frame for rate extraction. The exemplars (a) SPAN the frequency range — once-daily sleep=1
# up to high-frequency water=8 — so a base model doesn't plant a wrong magnitude prior on a need it
# hasn't seen (zero-shot gave sleep 3x/day, and the recluse MORE social than the baker — both wrong),
# and (b) vary the PERSON so person-dependence is reinforced, not flattened (baker social 8 vs hermit
# 1). Verified: few-shot fixes both the magnitude AND the per-person spread vs zero-shot.
RATE_FEWSHOT = (
    "Person: a weary laborer at the end of a long day\n"
    "On a typical day, how many times does this person satisfy their sleep need?\nAnswer: 1\n\n"
    "Person: an active child running around outside\n"
    "On a typical day, how many times does this person satisfy their water need?\nAnswer: 8\n\n"
    "Person: a hardworking farmhand\n"
    "On a typical day, how many times does this person satisfy their food need?\nAnswer: 4\n\n"
    "Person: a content hermit who treasures solitude\n"
    "On a typical day, how many times does this person satisfy their social need?\nAnswer: 1\n\n"
    "Person: {person}\n"
    "On a typical day, how many times does this person satisfy their {need} need?\nAnswer:"
)


def rate_prompt(person, need):
    """`person` is a description string — the richer it is, the more individuated the rate. Few-shot
    (RATE_FEWSHOT): exemplars span the frequency range (sleep 1 .. water 8) so the model doesn't carry
    a wrong magnitude prior onto an unseen need, and vary the person so individuation is reinforced."""
    return RATE_FEWSHOT.format(person=person, need=need)


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


# --- durations: how long the ACTION of using an item takes (gen_duration -> minutes), baked PER
#     ITEM. Durations belong to actions, not needs: "eating a loaf of bread" is minutes, "sleeping on
#     a bed" is hours; how much each action RECOVERS the need is the affordance refill already baked.
#     The model can only time an action when it's named as a gerund EVENT ("eating a loaf of bread") —
#     a bare object ("a loaf of bread") yields near-zero/bimodal junk. So we NAME the action from
#     (item, need) first, then time it; gen_duration's median absorbs the occasional wild outlier.

def action_prompt(species, item, need):
    return (f"A {species} uses a {item} to satisfy their {need} need.\n"
            f"In two or three words, the action they are doing is:")


def duration_for_action_prompt(species, action):
    return f"How long does {action} usually take for a {species}?\nAnswer:"


def bake_durations(server, species, affordance_table, samples=7, floor_min=0.5):
    """Per-item action duration in MINUTES: {item: minutes}. For each item, first NAME the gerund
    action from item + the need it serves (its primary/highest-refill need), then time that action —
    a bare object can't be timed stably. floor_min keeps an action committable (a 0-length activity is
    instantaneous, which reintroduces frantic topping-up). Items serving no need are skipped."""
    out = {}
    for item, served in affordance_table.items():
        if not served:
            continue
        need = max(served, key=served.get)                 # the item's primary activity
        action = server.gen_text(action_prompt(species, item, need), stop=["\n", "."], n_predict=12)
        if not action:
            continue
        r = server.gen_duration(duration_for_action_prompt(species, action), samples=samples,
                                check_subject=action)   # yes/no-validate the median, resample if not
        out[item] = max(round(r["minutes"], 1), floor_min) if r else None
    return out


# --- thresholds: the fullness level at which a need DEMANDS attention — the deadband the colony agent
#     acts on (seek the need once its level drops below this). Per-need. Few-shot (range-spanning +
#     question-repeated) because zero-shot was extreme/inconsistent (hygiene 1%, sleep 60%). The
#     extracted profile is sensible: survival needs come out PROACTIVE (water/health/safety ~0.8-0.9),
#     tolerable ones DEFERRED (hygiene 0.15, sleep 0.25). Returned as a fraction in [0,1].

THRESHOLD_FEWSHOT = (
    "A person's needs slowly run low. At what fullness do they stop and take care of each one? "
    "(100 = fully satisfied, 0 = empty — lower means they tolerate it longer.)\n\n"
    "Need: hygiene\nAt what percent full do they deal with it?\nAnswer: 15\n\n"
    "Need: sleep\nAt what percent full do they deal with it?\nAnswer: 25\n\n"
    "Need: hunger\nAt what percent full do they deal with it?\nAnswer: 35\n\n"
    "Need: thirst\nAt what percent full do they deal with it?\nAnswer: 50\n\n"
    "Need: safety\nAt what percent full do they deal with it?\nAnswer: 80\n\n"
    "Need: {need}\nAt what percent full do they deal with it?\nAnswer:"
)


def threshold_prompt(need):
    return THRESHOLD_FEWSHOT.format(need=need)


def bake_thresholds(server, need_list, samples=7, floor=0.05, cap=0.95):
    """Per-need deadband: {need: fullness-fraction at which the agent stops to satisfy it}. Median over
    the range-spanning, question-repeated few-shot (zero-shot was extreme: hygiene 1%, sleep 60%).
    Clamped to [floor, cap] so a need always eventually triggers yet never demands a literally-full bar."""
    out = {}
    for need in need_list:
        r = server.gen_number_median(threshold_prompt(need), samples=samples)
        v = (r["median"] / 100.0) if r else 0.35
        out[need] = round(min(max(v, floor), cap), 2)
    return out
