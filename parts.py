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

# diff: distinctive parts beyond the typical plan. Framing demands concrete NOUN parts ("name parts like a
# mane or tusks, not descriptions") — without it, a creature with an adjective-heavy description echoes the
# adjectives (Makit -> "smooth-skinned, porous") instead of finding a part.
DIFF_FEWSHOT = (
    "Question: What distinctive physical parts does a lion have that a typical mammal lacks? Name parts (like a mane or tusks), not descriptions.\nAnswer: mane\n"
    "Question: What distinctive physical parts does an elephant have that a typical mammal lacks? Name parts (like a mane or tusks), not descriptions.\nAnswer: tusks, trunk\n"
    "Question: What distinctive physical parts does a deer have that a typical mammal lacks? Name parts (like a mane or tusks), not descriptions.\nAnswer: antlers\n")
# verify (adversarial) — framed as "IS X a real body part" so a quality/adjective ("porous") is rejected,
# not just "does it have X" (which an adjective passes). Mixed Yes/No, high-perplexity order (Y N Y N -> no:
# Y N N Y), varied species.
VERIFY_PART_FEWSHOT = (
    "Question: Is a mane a real physical body part of a lion?\nAnswer: Yes\n"
    "Question: Is porous a real physical body part of a mushroom?\nAnswer: No\n"
    "Question: Is fluffy a real physical body part of a rabbit?\nAnswer: No\n"
    "Question: Are tusks a real physical body part of an elephant?\nAnswer: Yes\n")
# prune: per-species removal of template parts the species lacks (octopus->no shell, snake->no claws).
# Framed as anatomical PRESENCE — "is X a part of a {species}?". Earlier framings mis-fired: "have meat"
# read as possess-food (pruned a lion's meat); "harvest, do you get X" conflated absence with harvest-
# typicality (pruned a peacock's heart, present but not usually taken). "Is a part of" tests pure presence.
PRUNE_FEWSHOT = (
    "Question: Is meat a part of a deer?\nAnswer: Yes\n"
    "Question: Is a shell a part of an octopus?\nAnswer: No\n"
    "Question: Is a beak a part of a rabbit?\nAnswer: No\n"
    "Question: Is bone a part of a snake?\nAnswer: Yes\n")
PRUNE_KEEP = 0.3
NULL_TOKENS = {"none", "nothing", "n/a", "na", "unknown", "nothing unusual", "no", "nil"}


def classify_body_plan(server, species, desc=None):
    ctx = f"{species} is {desc}.\n" if desc else ""
    probs = server.gen_categorical(PLAN_FEWSHOT + ctx + f"What body plan does a {species} have? {_PO}\nAnswer:",
                                   [o for o, _ in PLAN_OPTS])
    o2c = dict(PLAN_OPTS)
    top = max(probs, key=probs.get)
    return o2c[top], round(probs[top], 3)


def species_part_diff(server, species, plan, desc=None, samples=3, threshold=0.5):
    """Distinctive parts beyond the body-plan template: multi-sample union (recall) -> adversarial verify
    (rejects qualities/adjectives, not just nonexistent parts)."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    prompt = (ctx + DIFF_FEWSHOT + f"Question: What distinctive physical parts does a {species} have that a "
              f"typical {plan} lacks? Name parts (like a mane or tusks), not descriptions.\nAnswer:")
    seen, cand = set(), []
    for _ in range(samples):
        for t in server.gen_text(prompt, stop=["\n"], n_predict=30).split(","):
            t = t.strip().lower().rstrip(".")
            if t and t not in seen and t not in NULL_TOKENS and len(t.split()) <= 3:
                seen.add(t); cand.append(t)
    kept = []
    for part in cand:
        if server.yes_no_prob(ctx + VERIFY_PART_FEWSHOT +
                              f"Question: Is {part} a real physical body part of a {species}?\nAnswer:") >= threshold:
            kept.append(part)
    return kept


def prune_prompt(species, part, desc=None):
    ctx = f"{species} is {desc}.\n" if desc else ""
    return ctx + PRUNE_FEWSHOT + f"Question: Is {part} a part of a {species}?\nAnswer:"


def prune_template(server, species, template, desc=None):
    """Drop template parts the species lacks (octopus has no shell). Conservative: keep unless the model is
    fairly sure it's absent (P(has) < PRUNE_KEEP)."""
    return [p for p in template if server.yes_no_prob(prune_prompt(species, p, desc)) >= PRUNE_KEEP]


def embed_dedup(items, seed=None, sim=0.82, url=bmp.EMBED_URL):
    """Drop near-duplicate strings by embedding cosine (peacock tail/plumage/train-plumage -> one). `seed`
    are kept-but-unemitted anchors (the template), so distinctive parts also dedup against the template."""
    anchors = list(bmp.embed_texts(seed, url)) if seed else []
    out = []
    for it in items:
        e = bmp.embed_texts([it], url)[0]
        if all(bmp.cosine(e, a) < sim for a in anchors):
            out.append(it); anchors.append(e)
    return out


def parts(server, species, desc=None, prune=True):
    """{plan, parts, distinctive}: body-plan template (per-species pruned) + verified, semantically-deduped
    species-specific distinctive parts."""
    plan, conf = classify_body_plan(server, species, desc)
    template = BODY_PLANS.get(plan, [])
    if prune:
        template = prune_template(server, species, template, desc)
    raw = [d for d in species_part_diff(server, species, plan, desc) if d not in set(template)]
    distinctive = embed_dedup(raw)   # dedup WITHIN the diff only (against-template ate Oak's 'acorns' ~ 'seed')
    return {"plan": plan, "plan_conf": conf, "parts": template + distinctive, "distinctive": distinctive}
