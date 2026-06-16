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

# Few-shot for the core yes/no sweep. Teaches REASON-FROM-DESCRIPTION so invented names ("Makit") and
# disambiguated plants both work: each shot states "{species} is {desc}." then asks one need, mixed yes/no.
# Spans living vs non-living (rock->no), and the SAME plant answering differently across needs (fern sleep
# ->no but water->yes) so the model learns the need is judged against the description, not defaulted.
CORE_FEWSHOT = (
    "A wolf is a pack-hunting wild canine that roams northern forests.\n"
    "If a wolf were real, would it have a food need (something it must regularly satisfy)? Answer yes or no.\nAnswer: yes\n\n"
    "A granite boulder is a solid lump of cooled mineral rock.\n"
    "If a granite boulder were real, would it have a water need (something it must regularly satisfy)? Answer yes or no.\nAnswer: no\n\n"
    "A fern is a leafy green plant that spreads across the shaded forest floor.\n"
    "If a fern were real, would it have a sleep need (something it must regularly satisfy)? Answer yes or no.\nAnswer: no\n\n"
    "A fern is a leafy green plant that spreads across the shaded forest floor.\n"
    "If a fern were real, would it have a water need (something it must regularly satisfy)? Answer yes or no.\nAnswer: yes\n\n"
    "A honeybee is a small social insect that lives in a colony and gathers nectar.\n"
    "If a honeybee were real, would it have a social need (something it must regularly satisfy)? Answer yes or no.\nAnswer: yes\n\n"
)

# Few-shot for the species-specific extras: description + known needs -> more single-word needs. The two
# exemplars span an animal and a plant so the model proposes domain-appropriate extras (territory vs sunlight).
# The answer sits inline after the colon (parsed to the newline) — no forced "\n- " bullet (the dash was
# vestigial: iter_unique already stops at "\n" and LOCATION_GRAMMAR forbids "-").
EXTRA_FEWSHOT = (
    "A wolf is a pack-hunting wild canine that roams northern forests.\n"
    "A wolf has needs such as: food, water, sleep.\n"
    "Another need a wolf has is: territory\n\n"
    "A fern is a leafy green plant that spreads across the shaded forest floor.\n"
    "A fern has needs such as: water, health.\n"
    "Another need a fern has is: sunlight\n\n"
)


def need_applies_prompt(species, need, desc=None):
    # "if it were real" sidesteps the model refusing needs to fictional beings (it fixated on
    # "dragons aren't real"); the counterfactual is harmless for species that already are real.
    # An optional `desc` grounds invented names: a fantasy creature ("Makit") is unguessable from the
    # name alone (empty core sweep), so we state "{species} is {desc}." above the question as context.
    ctx = f"{species} is {desc}.\n" if desc else ""
    return (CORE_FEWSHOT + f"{ctx}If a {species} were real, would it have a {need} need "
            f"(something it must regularly satisfy)? Answer yes or no.\nAnswer:")


def extra_needs_prompt(species, have, desc=None):
    ctx = f"{species} is {desc}.\n" if desc else ""
    return (EXTRA_FEWSHOT + f"{ctx}A {species} has needs such as: {', '.join(have)}.\n"
            f"Another need a {species} has is:")


def discover_core(server, species, threshold=0.5, desc=None):
    """yes/no-sweep the whole UNIVERSAL_CORE against `species`; each entry keeps P(yes)."""
    out = []
    for need in UNIVERSAL_CORE:
        p = server.yes_no_prob(need_applies_prompt(species, need, desc))
        out.append({"need": need, "applies": p >= threshold, "p": round(p, 3)})
    return out


def discover_needs(server, species, n_extra=6, threshold=0.5, desc=None):
    """Full hybrid discovery: core sweep + dedup-suggested single-word species-specific extras.
    EXCLUDE_NEEDS are dropped from the returned have/extra (the prompt still sees them so the model
    doesn't re-suggest them; the core sweep still records them). `desc` grounds invented names."""
    core = discover_core(server, species, threshold, desc)
    have_all = [c["need"] for c in core if c["applies"]]
    have = [n for n in have_all if n not in EXCLUDE_NEEDS]
    # extras dedup against the FULL core (not just the applying ones) so a dropped-but-borderline
    # core word can't resurface as an "extra"; reject non-single-word / non-alphabetic junk / excludes.
    extra = [e.lower() for e in iter_unique(
        server, extra_needs_prompt(species, have_all, desc), n=n_extra, grammar=LOCATION_GRAMMAR,
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
# hasn't seen (zero-shot gave sleep 3x/day, the recluse MORE social than the baker — both wrong), and
# (b) vary the PERSON so person-dependence is reinforced, not flattened. The full question with the
# EXPLICIT person folded in is repeated each shot under Question:/Answer: — no "this person" (an
# indirect reference the model pays to resolve); this sharpened the spread (baker social 3->10).
RATE_FEWSHOT = (
    "Question: On a typical day, how many times does a weary laborer satisfy their sleep need?\nAnswer: 1\n"
    "Question: On a typical day, how many times does an active child running around outside satisfy their water need?\nAnswer: 8\n"
    "Question: On a typical day, how many times does a hardworking farmhand satisfy their food need?\nAnswer: 4\n"
    "Question: On a typical day, how many times does a content hermit who treasures solitude satisfy their social need?\nAnswer: 1\n"
    "Question: On a typical day, how many times does {person} satisfy their {need} need?\nAnswer:"
)


def rate_prompt(person, need):
    """`person` is a description string — the richer it is, the more individuated the rate. Few-shot
    (RATE_FEWSHOT): exemplars span the frequency range (sleep 1 .. water 8) so the model doesn't carry
    a wrong magnitude prior onto an unseen need, vary the person to reinforce individuation, and fold
    the explicit person into the repeated Question:/Answer: (no pronouns)."""
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

# Few-shot frame for the affordance gate (which needs an item can fill). Mixed Yes/No exemplars
# (non-overlapping with real items, person-generic) sharpen the clear cases (yes_mean 0.82->0.98) AND
# recover borderline true-positives the zero-shot gate wrongly rejected (fireplace/warmth 0.36->0.95).
AFFORDANCE_GATE_FEWSHOT = (
    "Question: Is a bowl of soup relevant to satisfying a person's food need?\nAnswer: Yes\n"
    "Question: Is a wall mirror relevant to satisfying a person's food need?\nAnswer: No\n"
    "Question: Is a hammock relevant to satisfying a person's sleep need?\nAnswer: Yes\n"
    "Question: Is a brick relevant to satisfying a person's water need?\nAnswer: No\n"
    "Question: Is a {obj} relevant to satisfying a {species}'s {need} need?\nAnswer:"
)


def affordance_applies_prompt(species, obj_kind, need):
    # "relevant to satisfying" beats "affect ... at all" (which biased to No, gated meal->food at
    # 0.32). Direct possessive ("a {species}'s {need} need" — no "their" pronoun) + few-shot (mixed
    # Yes/No): sharpens clear cases (yes_mean ->0.98) and recovers borderline affordances zero-shot
    # dropped (fireplace/warmth 0.36->0.95); calibrated 9/9.
    return AFFORDANCE_GATE_FEWSHOT.format(obj=obj_kind, species=species, need=need)


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


# --- ITEM affordance (SPECIES-AGNOSTIC): which needs an item can fill in principle, for whatever creature
#     has that need. Same two-step (gate + gen_percent degree), no species in the frame — a bed fills sleep,
#     a fire warmth, a wooden beam food (for wood-eaters). The species join (which of an item's needs a given
#     species actually has) is downstream and free. Species-specific quirks ride the species-as-target path.
ITEM_AFFORDANCE_FEWSHOT = (
    "Question: Is a bowl of soup relevant to satisfying a food need?\nAnswer: Yes\n"
    "Question: Is a wall mirror relevant to satisfying a food need?\nAnswer: No\n"
    "Question: Is a hammock relevant to satisfying a sleep need?\nAnswer: Yes\n"
    "Question: Is a brick relevant to satisfying a water need?\nAnswer: No\n"
    "Question: Is a campfire relevant to satisfying a warmth need?\nAnswer: Yes\n"
    "Question: Is a {item} relevant to satisfying a {need} need?\nAnswer:"
)


def item_affordance_applies_prompt(item, need):
    return ITEM_AFFORDANCE_FEWSHOT.format(item=item, need=need)


def item_affordance_amount_prompt(item, need):
    """Degree framed for gen_percent (no species/possessive — species-agnostic)."""
    return (f"Q: How much does using a {item} satisfy a {need} need?\n"
            f"A: It satisfies the {need} need")


def bake_item_affordance(server, item, need, gate_threshold=0.5):
    """One (item, need): gate (yes_no) then degree (gen_percent). {applies, p, refill}."""
    p = server.yes_no_prob(item_affordance_applies_prompt(item, need))
    if p < gate_threshold:
        return {"applies": False, "p": round(p, 3), "refill": 0.0}
    pct = server.gen_percent(item_affordance_amount_prompt(item, need))
    return {"applies": True, "p": round(p, 3), "refill": round(pct["value"], 3) if pct else 0.0}


def bake_item_affordances(server, item, need_list, gate_threshold=0.5):
    """{need: refill} for the needs an item fills (gate-passing, refill>0) — the item's intrinsic profile."""
    served = {}
    for need in need_list:
        r = bake_item_affordance(server, item, need, gate_threshold)
        if r["applies"] and r["refill"] > 0:
            served[need] = r["refill"]
    return served


# --- durations: how long using an item TAKES (gen_duration -> minutes), baked PER ITEM. We DON'T
#     name a free-form "action" (that step was hopelessly noisy — empty strings, "water need",
#     "mugging", rambling tails). The affordance gate already told us which NEED the item serves, and
#     the NEED anchors the activity for us: asking "how long does using a {item} to satisfy their
#     {sleep} need take?" yields a night, "{food}" a meal — no naming required. The elicitation is
#     FEW-SHOT (range-spanning exemplars, water 1m..bed 8h) which anchors the magnitude so apparatus/
#     prep items don't read as cooking-length (stovetop 135m->30m, loaf 256m->90m); gen_duration's
#     median over `samples` plus the few-shot sanity gate (validated against the item+need itself)
#     absorb the rest. Residual: genuinely ambiguous items (raw stew) can still read as cooking — the
#     gate flags those with a low p_makes_sense.

DURATION_FEWSHOT = (
    "Question: How long does a person spend using a glass of water to satisfy their water need?\nAnswer: 1 minute\n"
    "Question: How long does a person spend using a hot meal to satisfy their food need?\nAnswer: 20 minutes\n"
    "Question: How long does a person spend using a shower to satisfy their hygiene need?\nAnswer: 10 minutes\n"
    "Question: How long does a person spend using a bed to satisfy their sleep need?\nAnswer: 8 hours\n"
    "Question: How long does a person spend using a {item} to satisfy their {need} need?\nAnswer:"
)


def duration_prompt(species, item, need):
    """Few-shot duration question (the grammar still forces a `<number> <unit>` answer). The full
    question with the EXPLICIT item+need is repeated each exemplar — no pronouns. `species` is kept
    for the signature; the exemplars are person-generic since durations are largely species-agnostic
    (which NEEDS an item serves already captures species differences)."""
    return DURATION_FEWSHOT.format(item=item, need=need)


def bake_durations(server, species, affordance_table, samples=8, floor_min=0.5):
    """Per-item activity length in MINUTES: {item: minutes}. Times the item's PRIMARY (highest-refill)
    need directly — the need anchors the activity, so no free-form action-naming. floor_min keeps the
    activity committable (a 0-length activity is instantaneous, reintroducing frantic topping-up).
    Items serving no need are skipped."""
    out = {}
    for item, served in affordance_table.items():
        if not served:
            continue
        need = max(served, key=served.get)                 # the item's primary need = its activity
        r = server.gen_duration(duration_prompt(species, item, need), samples=samples,
                                check_subject=f"using a {item} to satisfy their {need} need")
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


# --- consumable: is the item used up (gone) after one use? PER SPECIES — whether use destroys the item
#     depends on WHO uses it: a wall persists when a person leans on it but is eaten by a wall-eating
#     monster (else the monster gets free food off an indestructible wall). The few-shot carries a
#     species-CONTRAST pair (termite/beam Yes vs person/beam No) to teach the species-dependence, and
#     mixed Yes/No to beat the base model's reflexive 'No'. Validated 10/10. Full question repeated each
#     exemplar — no pronouns.
CONSUMABLE_FEWSHOT = (
    "Question: After a person uses a loaf of bread once, is it used up and gone?\nAnswer: Yes\n"
    "Question: After a person uses a bed once, is it used up and gone?\nAnswer: No\n"
    "Question: After a termite uses a wooden beam once, is it used up and gone?\nAnswer: Yes\n"
    "Question: After a person uses a wooden beam once, is it used up and gone?\nAnswer: No\n"
    "Question: After a {species} uses a {item} once, is it used up and gone?\nAnswer:"
)


def consumable_prompt(species, item):
    return CONSUMABLE_FEWSHOT.format(species=species, item=item)


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


# --- need MODE: HOW a need is met, which decides how the sim must treat it. Five modes, each a
#     few-shot Y/N ("Is a person's {need} need mainly satisfied by {desc}?"); the need is assigned to
#     the mode with the highest P(yes), species-agnostic ("a person's ... need" — the mode is a
#     property of the NEED, identical for every species). Explicit need name every shot (no pronouns),
#     mixed Yes/No spanning the range. Prototyped across 18 needs: archetypes land 0.84-0.98 confident
#     (food->consume, sleep->restore, shelter/safety/warmth->ambient, social/love->social,
#     novelty/fun->experiential); genuinely cross-cutting needs (comfort, respect, rest) fall below
#     MODE_FLOOR and are flagged `unsure` for manual assignment rather than committed on noise.
#       consume      - use up / ingest an item (food, water)           -> walk, use once, item may vanish
#       restore      - a personal upkeep activity (sleep, hygiene)      -> walk, do it over a duration
#       ambient      - a condition of WHERE you are (shelter, warmth)   -> a provider's field; passive
#       social       - proximity to other PEOPLE (companionship, love)  -> near other agents, not items
#       experiential - doing something new/meaningful (novelty, fun)    -> varied activities; variety
NEED_MODES = {
    "consume":      ("eating, drinking, or swallowing something",
                     ["hunger", "thirst", "medicine"], ["sleep", "safety", "friendship"]),
    "restore":      ("spending time on a personal activity like sleeping, washing, or using a bathroom",
                     ["sleep", "cleanliness", "toileting"], ["hunger", "shelter", "friendship"]),
    "ambient":      ("simply being in a safe, sheltered, or warm place",
                     ["shelter", "safety", "warmth"], ["hunger", "friendship", "sleep"]),
    "social":       ("spending time with or near other people",
                     ["friendship", "love", "companionship"], ["hunger", "shelter", "novelty"]),
    "experiential": ("doing something new, enjoyable, or personally meaningful",
                     ["novelty", "fun", "creativity"], ["hunger", "shelter", "sleep"]),
}
MODE_FLOOR = 0.45   # top score below this -> the need fits no mode cleanly; flag for manual assignment


def mode_prompt(mode, need):
    """Few-shot Y/N for one mode: 'Is a person's {need} need mainly satisfied by {desc}?'. The full
    question with the EXPLICIT need is repeated each exemplar — no pronouns."""
    desc, yes, no = NEED_MODES[mode]
    shots = "".join(f"Question: Is a person's {n} need mainly satisfied by {desc}?\nAnswer: {a}\n"
                    for n, a in ([(y, "Yes") for y in yes] + [(x, "No") for x in no]))
    return shots + f"Question: Is a person's {need} need mainly satisfied by {desc}?\nAnswer:"


def classify_mode(server, need):
    """Classify a need into one of the five modes (species-agnostic). Runs a Y/N per mode and takes the
    argmax of P(yes). Returns {mode, conf, unsure, scores} — `unsure` True when the top score is below
    MODE_FLOOR (the need fits no mode cleanly; the caller should ask the user to assign it)."""
    scores = {m: round(server.yes_no_prob(mode_prompt(m, need)), 3) for m in NEED_MODES}
    best = max(scores, key=scores.get)
    return {"mode": best, "conf": scores[best], "unsure": scores[best] < MODE_FLOOR, "scores": scores}


# --- AMBIENT affordance: an ambient-mode need (shelter/safety/warmth) is met by being in a PROVIDER's
#     field, not by "using" an item. So the gate asks whether the item PROVIDES the condition to the area
#     around it — which excludes loosely-associated furniture the use-gate let in (a chair is *in* a
#     sheltered room but provides no shelter). Validated 9/9 vs the use-gate's 7/9: chair/shelter
#     0.92->0.13, chair/safety 0.73->0.16, while providers (roof/fireplace/wall/tower) sharpen to 0.96-0.99.
AMBIENT_PROVIDER_FEWSHOT = (
    "Question: Does a roof provide shelter to the area around it?\nAnswer: Yes\n"
    "Question: Does a coffee mug provide shelter to the area around it?\nAnswer: No\n"
    "Question: Does a campfire provide warmth to the area around it?\nAnswer: Yes\n"
    "Question: Does a bookshelf provide warmth to the area around it?\nAnswer: No\n"
    "Question: Does a {obj} provide {need} to the area around it?\nAnswer:"
)


def provider_applies_prompt(obj_kind, need):
    return AMBIENT_PROVIDER_FEWSHOT.format(obj=obj_kind, need=need)


def provider_amount_prompt(obj_kind, need):
    """Strength of the field (gen_percent): how fully being near the provider meets the need."""
    return (f"Q: How much does being near a {obj_kind} satisfy a person's {need} need?\n"
            f"A: It satisfies their {need} need")


# --- ambient RADIUS: how far the field reaches, in GRID CELLS (~1 meter/cell). Few-shot anchors the
#     magnitude (a campfire a few cells; a watchtower's safety much farther). gen_number_median, floor 1.
RADIUS_FEWSHOT = (
    "On a grid where each cell is about one meter:\n\n"
    "Question: Within how many cells does the warmth from a campfire reach?\nAnswer: 4\n"
    "Question: Within how many cells does the shelter from a tent reach?\nAnswer: 3\n"
    "Question: Within how many cells does the warmth from a fireplace reach?\nAnswer: 3\n"
    "Question: Within how many cells does the safety from a watchtower reach?\nAnswer: 15\n"
    "Question: Within how many cells does the {need} from a {item} reach?\nAnswer:"
)


def radius_prompt(item, need):
    return RADIUS_FEWSHOT.format(item=item, need=need)


# --- SPECIES as a target: another creature can satisfy a consumer's need the same way an item can — the
#     gate is mode-appropriate (a CONSUME need is met by EATING the target, possibly killing it; a SOCIAL
#     or ambient need by being NEAR it). Validated: monster/human eat 0.99, person/person eat 0.06;
#     person/person near 0.98, person/rock near 0.02. consumable is per (consumer,target): eating usually
#     kills (wolf/rabbit) but not always (a vampire's victim survives).
SPECIES_EAT_FEWSHOT = (
    "Question: Can a wolf satisfy its food need by eating a rabbit?\nAnswer: Yes\n"
    "Question: Can a person satisfy its food need by eating another person?\nAnswer: No\n"
    "Question: Can a fox satisfy its food need by eating a chicken?\nAnswer: Yes\n"
    "Question: Can a deer satisfy its food need by eating a wolf?\nAnswer: No\n"
    "Question: Can a {consumer} satisfy its {need} need by eating a {target}?\nAnswer:"
)
SPECIES_NEAR_FEWSHOT = (
    "Question: Can a person satisfy its social need by spending time with another person?\nAnswer: Yes\n"
    "Question: Can a person satisfy its social need by spending time with a chair?\nAnswer: No\n"
    "Question: Can a dog satisfy its social need by spending time with a person?\nAnswer: Yes\n"
    "Question: Can a person satisfy its social need by spending time with a rock?\nAnswer: No\n"
    "Question: Can a {consumer} satisfy its {need} need by spending time with a {target}?\nAnswer:"
)
SPECIES_KILL_FEWSHOT = (
    "Question: After a wolf eats a rabbit once, is the rabbit dead and gone?\nAnswer: Yes\n"
    "Question: After a person spends time with a friend once, is the friend dead and gone?\nAnswer: No\n"
    "Question: After a bear eats a salmon once, is the salmon dead and gone?\nAnswer: Yes\n"
    "Question: After a mosquito bites a person once, is the person dead and gone?\nAnswer: No\n"
    "Question: After a {consumer} feeds on a {target} once, is the {target} dead and gone?\nAnswer:"
)


def species_affordance_prompt(mode, consumer, target, need):
    """Mode-appropriate gate: CONSUME -> eat the target, else (social/ambient) -> be near it."""
    tmpl = SPECIES_EAT_FEWSHOT if mode == "consume" else SPECIES_NEAR_FEWSHOT
    return tmpl.format(consumer=consumer, target=target, need=need)


def species_amount_prompt(mode, consumer, target, need):
    verb = "eating" if mode == "consume" else "being near"
    return (f"Q: How much does {verb} a {target} satisfy a {consumer}'s {need} need?\n"
            f"A: It satisfies their {need} need")


def species_kill_prompt(consumer, target):
    return SPECIES_KILL_FEWSHOT.format(consumer=consumer, target=target)


def bake_species_affordance(server, consumer, target, need, mode, gate_threshold=0.5):
    """Does target-species satisfy consumer-species' `need` (mode-appropriate gate + degree)?
    Returns {applies, p, refill}."""
    p = server.yes_no_prob(species_affordance_prompt(mode, consumer, target, need))
    if p < gate_threshold:
        return {"applies": False, "p": round(p, 3), "refill": 0.0}
    pct = server.gen_percent(species_amount_prompt(mode, consumer, target, need))
    return {"applies": True, "p": round(p, 3), "refill": round(pct["value"], 3) if pct else 0.0}


def bake_provider(server, obj_kind, need, gate_threshold=0.5, samples=5, radius_floor=1):
    """One ambient (item, need): provider-gate, then field strength (gen_percent) + radius (cells).
    Returns {applies, p_applies, strength, radius} — strength/radius 0 when the gate fails (species-
    agnostic: a physical field exists regardless of who stands in it)."""
    p = server.yes_no_prob(provider_applies_prompt(obj_kind, need))
    if p < gate_threshold:
        return {"applies": False, "p_applies": round(p, 3), "strength": 0.0, "radius": 0}
    pct = server.gen_percent(provider_amount_prompt(obj_kind, need))
    rr = server.gen_number_median(radius_prompt(obj_kind, need), samples=samples)
    return {"applies": True, "p_applies": round(p, 3),
            "strength": round(pct["value"], 3) if pct else 0.0,
            "radius": max(radius_floor, round(rr["median"])) if rr else 3}


# --- DIET-TYPE: a species trait that gates EATING affordances — a carnivore won't eat plant-food, and a
#     photosynthetic plant eats NOTHING (its "food" need is met by its own sunlight/soil/nutrient needs).
#     Read as a DISTRIBUTION over options (gen_categorical), not a sample: the argmax is more reliable
#     (Loquat sampled 'omnivore' but the distribution is sunlight 0.83) and the top prob is a confidence/
#     ambiguity flag. Plants/Fungi are assigned from KINGDOM (definitional — no query, no error).
DIET_OPTS = [("meat", "carnivore"), ("plants", "herbivore"), ("both", "omnivore"),
             ("sunlight", "photosynthetic"), ("dead matter", "decomposer"), ("other", "other")]
_DIET_O = "(meat, plants, both, sunlight, dead matter, or other)"
DIET_FEWSHOT = (
    f"What does a lion mainly eat? {_DIET_O}\nAnswer: meat\n"
    f"What does a rabbit mainly eat? {_DIET_O}\nAnswer: plants\n"
    f"What does a raccoon mainly eat? {_DIET_O}\nAnswer: both\n"
    f"What does an oak tree mainly eat? {_DIET_O}\nAnswer: sunlight\n"
    f"What does a mushroom mainly eat? {_DIET_O}\nAnswer: dead matter\n")
DIET_FLOOR = 0.45  # top prob below this -> cross-cutting/exotic diet, flag unsure (route to 'other')


def diet_prompt(species, desc=None):
    ctx = f"{species} is {desc}.\n" if desc else ""
    return DIET_FEWSHOT + f"{ctx}What does a {species} mainly eat? {_DIET_O}\nAnswer:"


def classify_diet(server, species, desc=None):
    """Diet-type via the gen_categorical distribution. {diet, conf, unsure, dist}. argmax of the
    distribution (beats a sample); unsure when top prob < DIET_FLOOR."""
    probs = server.gen_categorical(diet_prompt(species, desc), [o for o, _ in DIET_OPTS])
    o2c = dict(DIET_OPTS)
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    top, conf = ranked[0]
    return {"diet": o2c[top], "conf": round(conf, 3), "unsure": conf < DIET_FLOOR,
            "dist": {o2c[o]: round(p, 3) for o, p in ranked if p >= 0.02}}
