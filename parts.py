"""
parts.py — partonomy: the harvestable PARTS of an entity (the 'drops' when butchered/harvested), the
part-whole dual of categories.py's taxonomy. Spine = hand-authored BODY-PLAN TEMPLATES at HARVEST
granularity (a distinct usable resource — meat/hide/bone/tusk, not protein complexes), selected by a
gen_categorical type classifier; then a generative SPECIES-DIFF (multi-sample union + adversarial verify)
adds the distinctive parts (mane, tusks, antlers). Inherit-defaults-store-diffs, like needs (core+extras)
and the exception layer. (Per-species pruning of inapplicable template parts is a later refinement.)
"""
import baseModelPrimitives as bmp

# Templates = the COMMON full-body inventory per plan, bootstrapped via sample_union (a broad prompt spanning
# EXTERNAL parts / ORGANS / TISSUES / SUBSTANCES) then curated; grouped by layer below. Species-specific parts
# (horns, antlers, tusks, wool, hooves, mane, fingers) live in the species-diff, not here; pruning drops any
# template part a given species lacks (a snake's legs, an octopus's shell).
BODY_PLANS = {
    "mammal": [
        "head", "neck", "legs", "paws", "toes", "tail", "ears", "eyes", "eyelids", "eyelashes", "eyebrows", "nose",
        "mouth", "lips", "gums", "teeth", "tongue", "whiskers", "claws", "nipples", "belly button",
        "meat", "hide", "fur", "hair", "bone", "blood", "fat", "sinew", "marrow",
        "heart", "liver", "lungs", "brain", "stomach", "intestines", "kidneys", "bladder", "spleen", "pancreas",
        "fingernails", "toenails", "saliva", "mucus", "earwax", "sweat", "tears", "urine", "droppings", "milk", "stomach acid"],
    "bird": [
        "head", "beak", "neck", "wings", "legs", "feet", "talons", "tail", "eyes",
        "feathers", "meat", "skin", "bone", "blood", "fat",
        "heart", "liver", "lungs", "gizzard", "crop", "kidneys", "intestines", "brain",
        "egg", "saliva", "mucus", "droppings", "stomach acid"],
    "reptile": [
        "head", "jaws", "tail", "legs", "feet", "claws", "eyes", "nostrils", "mouth", "tongue", "teeth",
        "scales", "skin", "meat", "bone", "blood", "fat",
        "heart", "liver", "lungs", "kidneys", "stomach", "brain",
        "eggs", "saliva", "mucus", "droppings", "stomach acid"],
    "amphibian": [
        "head", "legs", "toes", "eyes", "mouth", "tongue", "teeth",
        "skin", "meat", "bone", "blood", "fat",
        "heart", "liver", "lungs", "kidneys", "brain", "stomach",
        "eggs", "mucus", "slime", "stomach acid"],
    "fish": [
        "head", "fins", "tail", "scales", "gills", "eyes", "mouth", "teeth",
        "meat", "skin", "bone", "cartilage", "blood", "fat",
        "heart", "liver", "kidneys", "stomach", "swim bladder", "brain",
        "roe", "bile", "mucus", "slime", "stomach acid"],
    "insect": [
        "head", "thorax", "abdomen", "legs", "wings", "antennae", "mandibles", "compound eyes", "mouthparts",
        "exoskeleton", "chitin",
        "gut", "heart",
        "hemolymph", "eggs"],
    "mollusc": [
        "head", "foot", "mantle", "shell", "eyes", "mouth", "radula", "tentacles", "siphon", "gills",
        "meat", "blood",
        "heart", "liver", "kidney", "gonads",
        "ink", "mucus", "slime"],
    "plant": [
        "roots", "stem", "trunk", "branches", "bark", "leaves", "buds", "flowers", "fruit", "seeds",
        "pollen", "nectar", "sap", "resin", "wood"],
    "fungus": [
        "cap", "stem", "gills", "pores", "ring", "volva",
        "spores", "mycelium", "hyphae", "fruiting body", "flesh"],
    "mineral": ["ore", "crystal", "facets", "veins", "dust", "core"],
    "construct": [
        "casing", "chassis", "frame", "plating", "joints",
        "wiring", "cables", "circuits", "processor", "core", "gears", "motors", "actuators", "sensors",
        "battery", "fuel"],
}
PLAN_OPTS = [("mammal", "mammal"), ("bird", "bird"), ("reptile", "reptile"), ("amphibian", "amphibian"),
             ("fish", "fish"), ("insect", "insect"), ("mollusc", "mollusc"), ("plant", "plant"),
             ("fungus", "fungus"), ("mineral", "mineral"), ("machine", "construct")]
_PO = "(mammal, bird, reptile, amphibian, fish, insect, mollusc, plant, fungus, mineral, or machine)"
PLAN_FEWSHOT = (
    f"What body plan does a wolf have? {_PO}\nAnswer: mammal\n"
    f"What body plan does an octopus have? {_PO}\nAnswer: mollusc\n"
    f"What body plan does an oak tree have? {_PO}\nAnswer: plant\n"
    f"What body plan does a frog have? {_PO}\nAnswer: amphibian\n"
    f"What body plan does a beetle have? {_PO}\nAnswer: insect\n"
    f"What body plan does a salmon have? {_PO}\nAnswer: fish\n"
    f"What body plan does a robot have? {_PO}\nAnswer: machine\n")

# diff: distinctive parts beyond the typical plan. Framing demands concrete NOUN parts ("name parts like a
# mane or tusks, not descriptions") — without it, a creature with an adjective-heavy description echoes the
# adjectives (Makit -> "smooth-skinned, porous") instead of finding a part. Exemplars SPAN body-plans
# (bird/mammal/fish), not all-mammal, so the few-shot doesn't bias toward mammalian parts or under-prime
# other plans; answer-lengths mixed 1/2/1 (the multi in the middle).
DIFF_FEWSHOT = (
    "Question: What distinctive physical parts does a peacock have that a typical bird lacks? Name parts (like a mane or tusks), not descriptions.\nAnswer: crest\n"
    "Question: What distinctive physical parts does an elephant have that a typical mammal lacks? Name parts (like a mane or tusks), not descriptions.\nAnswer: tusks, trunk\n"
    "Question: What distinctive physical parts does a catfish have that a typical fish lacks? Name parts (like a mane or tusks), not descriptions.\nAnswer: barbels\n")
# verify (adversarial) — "would X be a real body part of it" so a quality/adjective ("porous") is rejected
# (not just "does it have X", which an adjective passes), under the COUNTERFACTUAL so a mythical creature's
# real part (a dragon's wings) isn't rejected as "not real". Mixed Yes/No, high-perplexity order (Y N N Y).
VERIFY_PART_FEWSHOT = (
    "Question: If a lion were real, would a mane be a real body part of it?\nAnswer: Yes\n"
    "Question: If a mushroom were real, would porous be a real body part of it?\nAnswer: No\n"
    "Question: If a rabbit were real, would fluffy be a real body part of it?\nAnswer: No\n"
    "Question: If an elephant were real, would tusks be a real body part of it?\nAnswer: Yes\n")
# prune: per-species removal of template parts the species lacks (octopus->no shell, snake->no claws).
# "would it HAVE {part}" under the COUNTERFACTUAL "if it were real". The counterfactual dodges the "isn't
# real" reflex that stripped dragons' organs (dragon liver 0.95). "have" (vs "be a part of it") also covers
# SUBSTANCES — earwax/mucus/sweat are had/produced but aren't "parts", so the part-framing wrongly pruned a
# lion's secretions; a substance exemplar (cat/earwax) teaches they count. With the right framing the probs
# are decisive (keep 0.9+, absent 0.02), so the threshold is the natural Y/N 0.5 — never tune the threshold
# to paper over a framing that's measuring the wrong predicate. Mixed Yes/No, high-perplexity order (Y N N Y).
PRUNE_FEWSHOT = (
    "Question: If a deer were real, would it have meat?\nAnswer: Yes\n"
    "Question: If an octopus were real, would it have a shell?\nAnswer: No\n"
    "Question: If a snake were real, would it have legs?\nAnswer: No\n"
    "Question: If a cat were real, would it have earwax?\nAnswer: Yes\n")
PRUNE_KEEP = 0.5
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
    cand = server.sample_union(prompt, samples=samples, n_predict=30, max_words=3,
                               reject=lambda k: k in NULL_TOKENS)
    kept = []
    for part in cand:
        if server.yes_no_prob(ctx + VERIFY_PART_FEWSHOT +
                              f"Question: If a {species} were real, would {part} be a real body part of it?\nAnswer:") >= threshold:
            kept.append(part)
    return kept


def prune_prompt(species, part, desc=None):
    ctx = f"{species} is {desc}.\n" if desc else ""
    return ctx + PRUNE_FEWSHOT + f"Question: If a {species} were real, would it have {part}?\nAnswer:"


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
