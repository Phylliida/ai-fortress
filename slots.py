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
MAX_SLOT_COUNT = 12     # clamp absurd counts (octopus "suckers: 3000", "armour: 64"); >12 wearable slots is impractical

# count: small-integer "how many X" (base models handle these fine; only big/fuzzy numbers confabulate).
# Mixed answers spanning 1..8, varied creatures/parts.
COUNT_FEWSHOT = (
    "How many eyes does a human have?\nAnswer: 2\n"
    "How many legs does a spider have?\nAnswer: 8\n"
    "How many tongues does a dog have?\nAnswer: 1\n"
    "How many arms does an octopus have?\nAnswer: 8\n")

# the spots we already cover — passed AS CONTEXT so the generator names only NEW ones (known-slots trick, same
# as the parts diff). Joined once, reused in the few-shot + query.
KNOWN_SITES = ", ".join(dict.fromkeys(s["site"] for s in HUMAN_SLOTS))

# extras: spots a species has BEYOND the ones it already covers (listed in-prompt). POSITIVE framing, multi-
# answer varying counts; "none" so human-like species add nothing. EXTRA_VERIFY backstops the human-shared
# parts the context can't catch (shoulders/hair/knees aren't in KNOWN_SITES, so the model may still leak them).
_EXTRA_Q = (lambda sp: f"A {sp} has body parts like {KNOWN_SITES}. Beyond those, what EXTRA body parts does a "
            f"{sp} have that you could wear or hang something on (like a tail, horns, or wings)? Name the body parts.")
SLOT_EXTRA_FEWSHOT = (
    f"{_EXTRA_Q('snake')}\nAnswer: none\n"
    f"{_EXTRA_Q('dragon')}\nAnswer: horns, wings, tail, snout\n"
    f"{_EXTRA_Q('centaur')}\nAnswer: horse back, hooves, horse tail\n")
# keep only spots a HUMAN LACKS (subtracts the shared parts the generator still leaks). Mixed Y/N, high-perplexity.
EXTRA_VERIFY_FEWSHOT = (
    "Question: Does a human have a tail?\nAnswer: No\n"
    "Question: Does a human have shoulders?\nAnswer: Yes\n"
    "Question: Does a human have wings?\nAnswer: No\n"
    "Question: Does a human have knees?\nAnswer: Yes\n")
_NUMWORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _strip_count(name):
    """Pull a leading count off an extra name: 'eight tentacles' -> ('tentacles', 8), 'horns' -> ('horns', None)."""
    w = name.split()
    if w and w[0].isdigit():
        return " ".join(w[1:]) or name, int(w[0])
    if w and w[0].lower() in _NUMWORDS:
        return " ".join(w[1:]) or name, _NUMWORDS[w[0].lower()]
    return name, None


def extract_count(server, species, site, desc=None):
    """Small integer for 'how many {site} does a {species} have?' (None if unparseable -> caller defaults)."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    raw = server.gen_text(ctx + COUNT_FEWSHOT + f"How many {site} does a {species} have?\nAnswer:",
                          stop=["\n"], n_predict=6)
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else None


def slot_extras(server, species, desc=None, samples=4):
    """Extra wearable body parts a species has beyond the ones it already covers (KNOWN_SITES given as context).
    sample-union -> embed-dedup -> TWO verifies: (a) it's a real body PART of the species (reuses
    parts.VERIFY_PART_FEWSHOT — rejects items like 'dagger'/'armour' and hallucinations like spider-wings), and
    (b) a HUMAN lacks it (subtracts shared parts like shoulders/hair the context can't catch)."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    raw = server.sample_union(ctx + SLOT_EXTRA_FEWSHOT + _EXTRA_Q(species) + "\nAnswer:",
                              samples=samples, n_predict=40, max_words=3, reject=lambda k: k in parts.NULL_TOKENS)
    out = []
    for e in parts.embed_dedup(raw):
        is_part = server.yes_no_prob(ctx + parts.VERIFY_PART_FEWSHOT +
                                     f"Question: If a {species} were real, would {e} be a real body part of it?\nAnswer:") >= 0.5
        human_has = server.yes_no_prob(EXTRA_VERIFY_FEWSHOT + f"Question: Does a human have {e}?\nAnswer:") >= 0.5
        if is_part and not human_has:
            out.append(e)
    return out


def species_slots(server, species, desc=None):
    """Per-species paper-doll: surviving HUMAN_SLOTS (site pruned) with per-species counts, plus extra slots.
    Returns a list of {slot, site, kind, count}."""
    sites = list(dict.fromkeys(s["site"] for s in HUMAN_SLOTS))
    have = {site: server.yes_no_prob(parts.prune_prompt(species, site, desc)) >= parts.PRUNE_KEEP for site in sites}
    clamp = lambda c: max(1, min(c, MAX_SLOT_COUNT))
    counts = {}
    for site in sites:
        if have[site] and site in MULTI_SITES:
            counts[site] = clamp(extract_count(server, species, site, desc) or MULTI_SITES[site])
    out = [{**s, "count": counts.get(s["site"], s["count"])} for s in HUMAN_SLOTS if have[s["site"]]]
    for e in slot_extras(server, species, desc):                # species-specific extra spots
        name, cnt = _strip_count(e)                             # 'eight tentacles' -> ('tentacles', 8)
        cnt = clamp(cnt or extract_count(server, species, name, desc) or 1)
        out.append({"slot": name, "site": name, "kind": "worn", "count": cnt})
    return out
