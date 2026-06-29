"""
loot.py — fill an entity's paper-doll (slots.py) with CONDITIONED gear, and (later) carried inventory.

Unlike anatomy, gear is CONTINGENT on WHO the entity is — species, occupation, wealth — so every generation is
conditioned on an entity descriptor ("A poor goblin raider"). Quality EMERGES from wealth ('tattered' vs
'silk'); empty slots come back as nothing (a beggar has no rings). Worn fill first, then the carried bag.
"""
import parts

# categorical wealth tiers (per the base-model "categories beat numbers" rule) — drive quality + how much is filled.
WEALTH_TIERS = ["destitute", "poor", "modest", "comfortable", "wealthy", "noble"]


def entity_prefix(entity):
    """'A poor goblin raider' / 'A wealthy elf merchant' from {wealth, gender, species, occupation}."""
    bits = [entity.get("wealth"), entity.get("gender"), entity.get("species", "person"), entity.get("occupation")]
    return "A " + " ".join(b for b in bits if b)


# presence gate: does this entity even USE this slot? Conditioned so LUXURIES (piercings, jewels, fancy
# carries) are wealth-gated, but BASICS (a torso covering, legwear, a combatant's weapon) survive even for the
# poor — a beggar still wears rags. Mixed Y/N, high-perplexity (Y N N Y Y N).
PRESENCE_FEWSHOT = (
    "Question: Does a destitute beggar wear something on their torso?\nAnswer: Yes\n"    # basic: even rags
    "Question: Does a poor goblin raider wear a nose ring?\nAnswer: No\n"                 # luxury: wealth-gated
    "Question: Does a destitute beggar wear rings on their fingers?\nAnswer: No\n"
    "Question: Is a poor goblin raider carrying a weapon or tool?\nAnswer: Yes\n"         # combatant: armed
    "Question: Does a wealthy merchant carry a satchel?\nAnswer: Yes\n"
    "Question: Does a poor farmer wear a gold necklace?\nAnswer: No\n")

# conditioned fill: wealth -> quality (silk vs threadbare), role -> appropriate gear. Anchors the drifty slots
# (belt/satchel/nose) with explicit examples so the item stays a thing-OF-that-slot, short noun phrase. Per the
# guidelines: EXPLICIT SUBJECT in the question (no 'they' pronoun), Question:/Answer: labels, mixed wealth/role/
# species/kind.
FILL_FEWSHOT = (
    "Question: What does a wealthy human noble wear on their torso?\nAnswer: an embroidered silk doublet\n"
    "Question: What does a destitute human beggar wear on their torso?\nAnswer: a filthy threadbare tunic\n"
    "Question: What does a poor goblin raider wear in their nose?\nAnswer: a crude bone nose-ring\n"
    "Question: What does a modest human farmer wear on their head?\nAnswer: a wide-brimmed straw hat\n"
    "Question: What does a wealthy elf merchant wear on their fingers?\nAnswer: a gold signet ring\n"
    "Question: What does a poor human soldier hold in their hands?\nAnswer: a notched iron shortsword\n"
    "Question: What does a modest human traveler wear on their belt?\nAnswer: a worn leather belt with a brass buckle\n"
    "Question: What does a wealthy human lady carry as their satchel?\nAnswer: a beaded velvet purse\n")


def _subj(entity):
    p = entity_prefix(entity)
    return p[0].lower() + p[1:]                          # 'a poor goblin raider' for in-question use


def _fill_q(subj, slot, kind):
    if kind == "held":
        return f"What does {subj} hold in their hands?"
    if kind == "container":
        return f"What does {subj} carry as their {slot}?"
    if kind in ("piercing", "jewelry"):
        return f"What does {subj} wear in their {slot}?"
    return f"What does {subj} wear on their {slot}?"


def has_slot(server, entity, slot, kind):
    """Presence gate — does this entity even USE this slot? Conditioned, so luxuries are wealth-gated but
    basics survive for the poor."""
    subj = _subj(entity)
    if kind == "held":
        q = f"Is {subj} carrying a weapon or tool?"
    elif kind == "container":
        q = f"Does {subj} carry a {slot}?"
    elif kind in ("piercing", "jewelry"):
        q = f"Does {subj} wear a {slot}?"
    else:
        q = f"Does {subj} wear something on their {slot}?"
    return server.yes_no_prob(PRESENCE_FEWSHOT + f"Question: {q}\nAnswer:") >= 0.5


def fill_slot(server, entity, slot, kind="worn", max_words=7):
    """The item in `slot` for this entity (or None if empty / a ramble), conditioned on who they are. Quality
    is carried by the wording the wealth tier elicits. Stops at a period and caps length so it stays a short
    noun phrase, not a sentence. Plurality rides the phrase ('silver rings'), so multi slots need no special case."""
    raw = server.gen_text(FILL_FEWSHOT + "Question: " + _fill_q(_subj(entity), slot, kind) + "\nAnswer:",
                          stop=["\n", "."], n_predict=14)
    item = raw.strip().strip(".,").strip()
    if not item or item.lower() in parts.NULL_TOKENS or len(item.split()) > max_words:
        return None
    return item


def fill_worn(server, entity, skeleton):
    """Dress every slot in the species `skeleton` (from slots.species_slots), conditioned on the entity:
    presence-gate each slot (poor -> sparse), then fill the ones that pass. Returns the FILLED slots as
    [{slot, kind, count, item}]."""
    out = []
    for s in skeleton:
        if not has_slot(server, entity, s["slot"], s["kind"]):
            continue
        item = fill_slot(server, entity, s["slot"], s["kind"])
        if item:
            out.append({"slot": s["slot"], "kind": s["kind"], "count": s["count"], "item": item})
    return out


def dress(server, entity, desc=None):
    """Full head-to-toe: build the species paper-doll, then fill it for this entity."""
    import slots
    skeleton = slots.species_slots(server, entity["species"], desc)
    return fill_worn(server, entity, skeleton)
