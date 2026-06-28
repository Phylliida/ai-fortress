"""
parts.py — partonomy: the harvestable PARTS of an entity (the 'drops' when butchered/harvested), the
part-whole dual of categories.py's taxonomy. Spine = hand-authored BODY-PLAN TEMPLATES at HARVEST
granularity (a distinct usable resource — meat/hide/bone/tusk, not protein complexes), selected by a
gen_categorical type classifier; then a generative SPECIES-DIFF (multi-sample union + adversarial verify)
adds the distinctive parts (mane, tusks, antlers). Inherit-defaults-store-diffs, like needs (core+extras)
and the exception layer. (Per-species pruning of inapplicable template parts is a later refinement.)
"""
import baseModelPrimitives as bmp

BODY_PLANS = {
    "mammal":   ["meat", "hide", "bone", "blood", "fat", "sinew", "heart", "liver", "lungs", "brain", "stomach", "intestines", "teeth"],
    "bird":     ["meat", "feathers", "bone", "blood", "fat", "heart", "liver", "gizzard", "beak", "egg", "talons"],
    "reptile":  ["meat", "scales", "bone", "blood", "heart", "liver", "egg", "claws"],
    "fish":     ["meat", "scales", "bone", "fins", "roe", "liver", "gills", "eyes"],
    "insect":   ["shell", "wings", "legs", "antennae", "mandibles"],
    "mollusc":  ["meat", "shell", "tentacles", "ink", "mantle"],
    "plant":    ["root", "stem", "leaves", "bark", "flower", "fruit", "seed", "sap", "wood"],
    "fungus":   ["cap", "stem", "spores", "mycelium", "gills"],
    "mineral":  ["ore", "crystal", "stone", "dust"],
    "construct": ["frame", "plating", "wiring", "core", "gears"],
}
PLAN_OPTS = [("mammal", "mammal"), ("bird", "bird"), ("reptile", "reptile"), ("fish", "fish"), ("insect", "insect"),
             ("mollusc", "mollusc"), ("plant", "plant"), ("fungus", "fungus"), ("mineral", "mineral"), ("machine", "construct")]
_PO = "(mammal, bird, reptile, fish, insect, mollusc, plant, fungus, mineral, or machine)"
PLAN_FEWSHOT = (
    f"What body plan does a wolf have? {_PO}\nAnswer: mammal\n"
    f"What body plan does an octopus have? {_PO}\nAnswer: mollusc\n"
    f"What body plan does an oak tree have? {_PO}\nAnswer: plant\n"
    f"What body plan does a beetle have? {_PO}\nAnswer: insect\n"
    f"What body plan does a salmon have? {_PO}\nAnswer: fish\n"
    f"What body plan does a robot have? {_PO}\nAnswer: machine\n")

# diff: distinctive parts beyond the typical plan (lists, multi-sample union)
DIFF_FEWSHOT = (
    "Question: What notable body parts does a lion have that a typical mammal does not? Just the unusual ones.\nAnswer: mane\n"
    "Question: What notable body parts does an elephant have that a typical mammal does not? Just the unusual ones.\nAnswer: tusks, trunk\n"
    "Question: What notable body parts does a deer have that a typical mammal does not? Just the unusual ones.\nAnswer: antlers\n")
# verify (adversarial). mixed Yes/No in high-perplexity order (Y N N Y), varied species
VERIFY_PART_FEWSHOT = (
    "Question: Does a lion genuinely have a mane?\nAnswer: Yes\n"
    "Question: Does a rabbit genuinely have antlers?\nAnswer: No\n"
    "Question: Does a lion genuinely have feathers?\nAnswer: No\n"
    "Question: Does an elephant genuinely have tusks?\nAnswer: Yes\n")


def classify_body_plan(server, species, desc=None):
    ctx = f"{species} is {desc}.\n" if desc else ""
    probs = server.gen_categorical(PLAN_FEWSHOT + ctx + f"What body plan does a {species} have? {_PO}\nAnswer:",
                                   [o for o, _ in PLAN_OPTS])
    o2c = dict(PLAN_OPTS)
    top = max(probs, key=probs.get)
    return o2c[top], round(probs[top], 3)


def species_part_diff(server, species, plan, desc=None, samples=3, threshold=0.5):
    """Distinctive parts beyond the body-plan template: multi-sample union (recall) -> adversarial verify."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    prompt = (ctx + DIFF_FEWSHOT + f"Question: What notable body parts does a {species} have that a typical "
              f"{plan} does not? Just the unusual ones.\nAnswer:")
    seen, cand = set(), []
    for _ in range(samples):
        for t in server.gen_text(prompt, stop=["\n"], n_predict=30).split(","):
            t = t.strip().lower().rstrip(".")
            if t and t not in seen and len(t.split()) <= 3:
                seen.add(t); cand.append(t)
    kept = []
    for part in cand:
        if server.yes_no_prob(ctx + VERIFY_PART_FEWSHOT + f"Question: Does a {species} genuinely have {part}?\nAnswer:") >= threshold:
            kept.append(part)
    return kept


def parts(server, species, desc=None):
    """{plan, parts, distinctive}: body-plan template + verified species-specific distinctive parts."""
    plan, conf = classify_body_plan(server, species, desc)
    template = BODY_PLANS.get(plan, [])
    seen = set(template)
    distinctive = [d for d in species_part_diff(server, species, plan, desc) if d not in seen]
    return {"plan": plan, "plan_conf": conf, "parts": template + distinctive, "distinctive": distinctive}
