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
        "head", "neck", "arms", "legs", "hands", "feet", "paws", "fingers", "toes", "tail", "ears", "eyes", "eyelids", "eyelashes", "eyebrows",
        "nose", "nostrils", "mouth", "lips", "gums", "teeth", "tongue", "whiskers", "claws", "nipples",
        "belly button", "genitals", "anus", "eardrums",
        "meat", "hide", "skin", "fur", "hair", "bone", "ribs", "cartilage", "blood", "veins", "fat", "sinew", "marrow",
        "heart", "liver", "lungs", "brain", "spinal cord", "stomach", "intestines", "kidneys", "bladder",
        "spleen", "pancreas", "gallbladder",
        "fingernails", "toenails", "saliva", "mucus", "earwax", "sweat", "tears", "urine", "droppings", "milk",
        "stomach acid", "bile"],
    "bird": [
        "head", "beak", "neck", "wings", "legs", "feet", "talons", "claws", "tail", "eyes", "eardrums", "nostrils", "tongue",
        "feathers", "down", "meat", "skin", "bone", "marrow", "blood", "fat",
        "heart", "liver", "lungs", "trachea", "gizzard", "crop", "kidneys", "intestines", "spleen", "pancreas",
        "brain", "cloaca",
        "egg", "saliva", "mucus", "droppings", "stomach acid"],
    "reptile": [
        "head", "jaws", "tail", "legs", "feet", "claws", "eyes", "eardrums", "nostrils", "mouth", "tongue", "teeth",
        "scales", "skin", "meat", "bone", "blood", "fat",
        "heart", "liver", "lungs", "kidneys", "stomach", "intestines", "spleen", "pancreas", "bladder", "brain",
        "cloaca",
        "eggs", "saliva", "mucus", "droppings", "stomach acid"],
    "amphibian": [
        "head", "legs", "feet", "toes", "eyes", "eardrums", "nostrils", "mouth", "tongue", "teeth", "gills",
        "skin", "meat", "bone", "blood", "fat",
        "heart", "liver", "lungs", "kidneys", "spleen", "bladder", "brain", "spinal cord", "stomach", "cloaca",
        "eggs", "mucus", "slime", "stomach acid"],
    "fish": [
        "head", "fins", "tail", "scales", "gills", "eyes", "mouth", "teeth", "lateral line",
        "meat", "skin", "bone", "cartilage", "blood", "fat",
        "heart", "liver", "kidneys", "stomach", "intestines", "spleen", "pancreas", "gallbladder", "swim bladder",
        "brain", "gonads", "cloaca",
        "roe", "bile", "mucus", "slime", "stomach acid"],
    "insect": [
        "head", "thorax", "abdomen", "legs", "wings", "antennae", "mandibles", "compound eyes", "mouthparts",
        "exoskeleton", "chitin", "spiracles",
        "brain", "gut", "crop", "heart",
        "hemolymph", "eggs"],
    "mollusc": [
        "head", "foot", "mantle", "shell", "eyes", "mouth", "beak", "radula", "tentacles", "siphon", "gills",
        "meat", "skin", "blood",
        "brain", "heart", "liver", "kidney", "gonads", "visceral mass",
        "ink", "mucus", "slime", "anus"],
    "plant": [
        "roots", "stem", "trunk", "branches", "bark", "leaves", "buds", "flowers", "petals", "stamen", "pistil",
        "fruit", "seeds", "thorns",
        "pollen", "nectar", "sap", "resin", "wood"],
    "fungus": [
        "cap", "stem", "gills", "pores", "ring", "veil", "volva",
        "spores", "mycelium", "hyphae", "fruiting body", "flesh"],
    "mineral": ["ore", "crystal", "facets", "veins", "dust", "core"],
    "construct": [
        "casing", "chassis", "frame", "plating", "joints", "axles", "bolts", "springs", "pistons",
        "wiring", "cables", "circuits", "processor", "core", "gears", "motors", "actuators", "sensors",
        "valves", "switches", "pipes", "lights",
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

# diff: distinctive parts BEYOND what the template already covers. We list the species' ALREADY-INCLUDED parts
# (the pruned template) as context so the model names genuinely NEW parts (a mane), not covered ones. Under
# the COUNTERFACTUAL "if it were real" (like prune/verify) so a mythical creature's parts aren't suppressed as
# "not real". POSITIVE framing — "name only its NATURAL BODY PARTS" — scopes to anatomy, keeping out clothing
# and adjectives WITHOUT naming them (a base model latches onto a mentioned "don't X"); the verify backstops.
# Exemplar ANSWERS are multi-part with VARYING counts (2/4/3) so the model doesn't learn to emit just one;
# exemplars span plans (bird / fantasy-reptile / fish). The instruction anchors GRANULARITY positively — "at
# a similar level of detail as the parts listed" — so it matches the template's level (a mane, not the
# "sartorius muscle"; chromatophores, not a "paralarva" life-stage) without a negative "don't go too fine".
_DIFF_INSTR = "Name only its natural body parts, at a similar level of detail as the parts listed (like a mane, tusks, or pointed ears)"
DIFF_FEWSHOT = (
    f"If a peacock were real, it would have these parts: feathers, wings, beak, legs, tail, eyes, meat, bone. Besides those, what other distinctive body parts would it have? {_DIFF_INSTR}.\nAnswer: crest, long train\n"
    f"If a dragon were real, it would have these parts: head, legs, tail, scales, claws, teeth, heart, bone. Besides those, what other distinctive body parts would it have? {_DIFF_INSTR}.\nAnswer: wings, horns, spikes, tail frills\n"
    f"If a catfish were real, it would have these parts: fins, gills, scales, tail, eyes, mouth, meat, bone. Besides those, what other distinctive body parts would it have? {_DIFF_INSTR}.\nAnswer: barbels, adipose fin, fin spines\n")
# verify (adversarial) — "would X be a real body part of it" rejects a quality/adjective ("porous") AND
# CLOTHING / worn items ("a hat"), not just "does it have X" (an adjective/hat passes that). Under the
# COUNTERFACTUAL so a mythical creature's real part (a dragon's wings) isn't denied as "not real". Mixed
# Yes/No, high-perplexity order (Y N N Y).
VERIFY_PART_FEWSHOT = (
    "Question: If a lion were real, would a mane be a real body part of it?\nAnswer: Yes\n"
    "Question: If a wizard were real, would a hat be a real body part of it?\nAnswer: No\n"
    "Question: If a mushroom were real, would porous be a real body part of it?\nAnswer: No\n"
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


def species_part_diff(server, species, plan, desc=None, known_parts=None, samples=4, threshold=0.5):
    """Distinctive parts beyond what the template covers. Lists `known_parts` (the pruned template) as context
    so the model names genuinely NEW parts; multi-sample union (recall) -> adversarial verify (rejects
    qualities/adjectives and clothing)."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    listed = ", ".join(known_parts) if known_parts else f"the usual {plan} parts"
    prompt = (ctx + DIFF_FEWSHOT + f"If a {species} were real, it would have these parts: {listed}. Besides those, "
              f"what other distinctive body parts would it have? {_DIFF_INSTR}.\nAnswer:")
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
    are kept-but-unemitted anchors (the template), so distinctive parts also dedup against the template.
    Falls back to exact (case-insensitive) dedup if the embed server is unreachable — so a flaky 8062 degrades
    dedup quality but never BLANKS a runtime result (a mimic's whole bag once vanished on an embed outage)."""
    try:
        anchors = list(bmp.embed_texts(seed, url)) if seed else []
        out = []
        for it in items:
            e = bmp.embed_texts([it], url)[0]
            if all(bmp.cosine(e, a) < sim for a in anchors):
                out.append(it); anchors.append(e)
        return out
    except Exception:                                      # embed down -> exact dedup, preserve order
        seen, out = set(), []
        for it in items:
            k = it.lower().strip()
            if k and k not in seen:
                seen.add(k); out.append(it)
        return out


# --- MACHINES: recursive decomposition instead of a template. Living things share a finite set of body plans
#     (templates win); machines are open-ended (a clock vs a server vs a mech), so we'd need a template per
#     TYPE — too many to be useful. Instead, ask a machine's high-level subsystems, then recurse each (with a
#     stop-gate at single pieces) into a part-tree. Same generate-few-then-recurse pattern, multi-answer union.
# Few-shot bootstrapped across DOMAINS (mechanical/food/biology/music) and DEPTHS (root/mid/deep). The ANCESTOR
# breadcrumb is carried in for two reasons: (1) it disambiguates a deep leaf — "ADP" under blood→platelets→
# activation is adenosine diphosphate, not the computing sense; (2) the "of the {leaf}" focus stops the model
# decomposing the ROOT instead of the leaf (a "car → engine" must give engine parts, not wheels/doors). Root
# nodes use the bare form; deeper nodes the "In {breadcrumb}, … of the {leaf}" form.
SUBPART_FEWSHOT = (
    "What are the main parts of a bicycle?\nAnswer: frame, wheels, pedals, chain, handlebars, seat\n"
    "In a car → engine, what are the main parts of the engine?\nAnswer: pistons, cylinders, crankshaft, valves, spark plugs\n"
    "In a pizza → topping, what are the main parts of the topping?\nAnswer: cheese, tomato sauce, pepperoni, herbs\n"
    "In blood → platelets → platelet activation → ADP, what are the main parts of the ADP?\nAnswer: adenosine, ribose, phosphate groups\n"
    "In a guitar → string, what are the main parts of the string?\nAnswer: core wire, winding, ball end\n")
# stop-gate: recurse only into parts that are themselves assemblies. Mixed Yes/No, high-perplexity (Y N N Y).
DECOMP_GATE_FEWSHOT = (
    "Question: Can an engine be taken apart into several smaller pieces?\nAnswer: Yes\n"
    "Question: Can a nail be taken apart into several smaller pieces?\nAnswer: No\n"
    "Question: Can a spring be taken apart into several smaller pieces?\nAnswer: No\n"
    "Question: Can a circuit board be taken apart into several smaller pieces?\nAnswer: Yes\n")
# per-part verify against the parent — filters derailed temp-1 draws (a "cyclotron" in a shock absorber).
# Counterfactual + "would it have" (consistent with the body-part prune); mixed Yes/No, high-perplexity (Y N N Y).
VERIFY_SUBPART_FEWSHOT = (
    "Question: If an engine were real, would it have a piston?\nAnswer: Yes\n"
    "Question: If a shock absorber were real, would it have a cyclotron?\nAnswer: No\n"
    "Question: If a bicycle were real, would it have a banana?\nAnswer: No\n"
    "Question: If a laptop were real, would it have a hard drive?\nAnswer: Yes\n")


def subpart_prompt(thing, ancestors=None):
    """Generation question for `thing` in the context of its `ancestors` (root-first). Root -> bare form;
    deeper -> 'In {breadcrumb}, what are the main parts of the {thing}?' (context + leaf-focus)."""
    if ancestors:
        return f"In {' → '.join(list(ancestors) + [thing])}, what are the main parts of the {thing}?"
    return f"What are the main parts of a {thing}?"


def machine_subparts(server, thing, ancestors=None, desc=None, samples=4, threshold=0.5):
    """Main parts of `thing` in the context of its `ancestors` breadcrumb, so an ambiguous deep leaf decomposes
    in context (ADP under blood→platelets = adenosine diphosphate, not the computing sense) and targets the leaf
    not the root. sample-union -> embed-dedup -> per-part VERIFY (kills derailed draws like a cyclotron)."""
    ctx = f"{thing} is {desc}.\n" if desc else ""
    raw = server.sample_union(ctx + SUBPART_FEWSHOT + subpart_prompt(thing, ancestors) + "\nAnswer:",
                              samples=samples, n_predict=50, max_words=3)
    return [p for p in embed_dedup(raw) if server.yes_no_prob(
        ctx + VERIFY_SUBPART_FEWSHOT + f"Question: If a {thing} were real, would it have a {p}?\nAnswer:") >= threshold]


def is_decomposable(server, part):
    """Is `part` itself an assembly (worth recursing into) vs a single piece?"""
    return server.yes_no_prob(DECOMP_GATE_FEWSHOT +
                              f"Question: Can a {part} be taken apart into several smaller pieces?\nAnswer:") >= 0.5


def decompose_machine(server, machine, desc=None, max_depth=2, samples=4):
    """Recursively decompose `machine` into a part-tree {part: subtree}. Stops at max_depth or single pieces.
    The colony-sim drops = every node; the leaves are the atomic salvage."""
    def rec(thing, ancestors, depth):
        subs = machine_subparts(server, thing, ancestors=ancestors, samples=samples)
        if depth <= 1:
            return {s: {} for s in subs}
        return {s: (rec(s, ancestors + [thing], depth - 1) if is_decomposable(server, s) else {}) for s in subs}
    return rec(machine, [], max_depth)


def _flatten_tree(tree):
    out = []
    for k, sub in tree.items():
        out.append(k)
        out.extend(_flatten_tree(sub))
    return out


def parts(server, species, desc=None, prune=True):
    """{plan, parts, ...}: living things get a body-plan template (pruned) + a verified species-diff; machines
    (construct) get a recursively-decomposed part-tree instead (no template — too many machine types)."""
    plan, conf = classify_body_plan(server, species, desc)
    if plan == "construct":
        tree = decompose_machine(server, species, desc)
        return {"plan": plan, "plan_conf": conf, "tree": tree, "parts": _flatten_tree(tree)}
    template = BODY_PLANS.get(plan, [])
    if prune:
        template = prune_template(server, species, template, desc)
    raw = [d for d in species_part_diff(server, species, plan, desc, known_parts=template) if d not in set(template)]
    distinctive = embed_dedup(raw)   # dedup WITHIN the diff only (against-template ate Oak's 'acorns' ~ 'seed')
    return {"plan": plan, "plan_conf": conf, "parts": template + distinctive, "distinctive": distinctive}
