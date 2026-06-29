"""
slots.py — per-species PAPER-DOLL skeleton: which wearable / carry locations an entity has and how many,
BEFORE we fill them with conditioned clothing & inventory.

Same four moves as parts.py's body-plan machinery, applied to wearable LOCATIONS instead of anatomy:
  HUMAN_SLOTS (master template) -> PRUNE (does the species have the body site?) -> COUNT (how many of a
  multiple site) -> EXTRAS (sites a species has beyond a human: tail, wings, horns). Reuses parts.prune_prompt
  (counterfactual presence) and parts.embed_dedup.

The output is a list of {slot, site, kind, count}: the empty skeleton later passes fill, one conditioned
clothing/inventory item per slot.
"""
import re
import parts

# Master human paper-doll. Each slot lives at a body SITE; the SITE is what we prune/count on (many slots
# share a site: shirt+coat→torso, gloves+held→hands, cloak+backpack→back). `kind` drives later FILLING
# (worn / held / container / piercing / jewelry). `count` = a human's default; only inherently-multiple
# sites (MULTI_SITES) get re-counted per species.
HUMAN_SLOTS = [
    {"slot": "hat",         "site": "head",     "kind": "worn",      "count": 1},
    {"slot": "glasses",     "site": "eyes",     "kind": "worn",      "count": 1},
    {"slot": "mask",        "site": "face",     "kind": "worn",      "count": 1},
    {"slot": "earring",     "site": "ears",     "kind": "piercing",  "count": 2},
    {"slot": "nose ring",   "site": "nose",     "kind": "piercing",  "count": 1},
    {"slot": "lip ring",    "site": "lips",     "kind": "piercing",  "count": 1},
    {"slot": "eyebrow ring","site": "eyebrows", "kind": "piercing",  "count": 1},
    {"slot": "tongue ring", "site": "tongue",   "kind": "piercing",  "count": 1},
    {"slot": "navel ring",  "site": "navel",    "kind": "piercing",  "count": 1},
    {"slot": "necklace",    "site": "neck",     "kind": "jewelry",   "count": 1},
    {"slot": "cloak",       "site": "back",     "kind": "worn",      "count": 1},
    {"slot": "bra",         "site": "chest",    "kind": "worn",      "count": 1},
    {"slot": "shirt",       "site": "torso",    "kind": "worn",      "count": 1},
    {"slot": "coat",        "site": "torso",    "kind": "worn",      "count": 1},
    {"slot": "belt",        "site": "waist",    "kind": "worn",      "count": 1},
    {"slot": "bracelet",    "site": "wrists",   "kind": "jewelry",   "count": 2},
    {"slot": "gloves",      "site": "hands",    "kind": "worn",      "count": 2},
    {"slot": "ring",        "site": "fingers",  "kind": "jewelry",   "count": 8},
    {"slot": "held",        "site": "hands",    "kind": "held",      "count": 2},
    {"slot": "underwear",   "site": "hips",     "kind": "worn",      "count": 1},
    {"slot": "pants",       "site": "legs",     "kind": "worn",      "count": 1},
    {"slot": "anklet",      "site": "ankles",   "kind": "jewelry",   "count": 2},
    {"slot": "socks",       "site": "feet",     "kind": "worn",      "count": 2},
    {"slot": "shoes",       "site": "feet",     "kind": "worn",      "count": 2},
    {"slot": "backpack",    "site": "back",     "kind": "container", "count": 1},
    {"slot": "satchel",     "site": "waist",    "kind": "container", "count": 1},
]

# sites that come in MULTIPLES (re-counted per species: a 4-armed mob → 4 gloves/held); value = human default.
MULTI_SITES = {"ears": 2, "hands": 2, "fingers": 8, "wrists": 2, "ankles": 2, "feet": 2}

# count: small-integer "how many X" (base models handle these fine; only big/fuzzy numbers confabulate).
# Mixed answers spanning 1..8, varied creatures/parts.
COUNT_FEWSHOT = (
    "How many eyes does a human have?\nAnswer: 2\n"
    "How many legs does a spider have?\nAnswer: 8\n"
    "How many tongues does a dog have?\nAnswer: 1\n"
    "How many arms does an octopus have?\nAnswer: 8\n")

# extras: wearable/hang spots a species has BEYOND a human. Counterfactual-free (real anatomy of the form),
# POSITIVE framing, multi-answer with varying counts; "none" exemplar so human-like species add nothing.
SLOT_EXTRA_FEWSHOT = (
    "Beyond a human's, what body spots does a snake have where you could wear or hang something? Name the spots.\nAnswer: none\n"
    "Beyond a human's, what body spots does a dragon have where you could wear or hang something? Name the spots.\nAnswer: horns, wings, tail, snout\n"
    "Beyond a human's, what body spots does a centaur have where you could wear or hang something? Name the spots.\nAnswer: horse back, four hooves, horse tail\n")


def extract_count(server, species, site, desc=None):
    """Small integer for 'how many {site} does a {species} have?' (None if unparseable -> caller defaults)."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    raw = server.gen_text(ctx + COUNT_FEWSHOT + f"How many {site} does a {species} have?\nAnswer:",
                          stop=["\n"], n_predict=6)
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else None


def slot_extras(server, species, desc=None, samples=4):
    """Wearable spots a species has beyond a human (tail/wings/horns) — sample-union, embed-deduped."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    raw = server.sample_union(
        ctx + SLOT_EXTRA_FEWSHOT +
        f"Beyond a human's, what body spots does a {species} have where you could wear or hang something? Name the spots.\nAnswer:",
        samples=samples, n_predict=40, max_words=3, reject=lambda k: k in parts.NULL_TOKENS)
    return parts.embed_dedup(raw)


def species_slots(server, species, desc=None):
    """Per-species paper-doll: surviving HUMAN_SLOTS (site pruned) with per-species counts, plus extra slots.
    Returns a list of {slot, site, kind, count}."""
    sites = list(dict.fromkeys(s["site"] for s in HUMAN_SLOTS))
    have = {site: server.yes_no_prob(parts.prune_prompt(species, site, desc)) >= parts.PRUNE_KEEP for site in sites}
    counts = {}
    for site in sites:
        if have[site] and site in MULTI_SITES:
            counts[site] = extract_count(server, species, site, desc) or MULTI_SITES[site]
    out = [{**s, "count": counts.get(s["site"], s["count"])} for s in HUMAN_SLOTS if have[s["site"]]]
    for e in slot_extras(server, species, desc):                # species-specific extra spots
        out.append({"slot": e, "site": e, "kind": "worn", "count": extract_count(server, species, e, desc) or 1})
    return out
