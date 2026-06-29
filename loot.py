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


# presence gate: does this entity even USE this slot? Conditioned so the POOR come out sparse (no fancy
# piercings) and the WEALTHY fuller. Mixed Y/N, high-perplexity (Y N N Y).
PRESENCE_FEWSHOT = (
    "Question: Does a wealthy noble wear something on their torso?\nAnswer: Yes\n"
    "Question: Does a poor goblin raider wear a nose ring?\nAnswer: No\n"
    "Question: Does a destitute beggar wear rings on their fingers?\nAnswer: No\n"
    "Question: Does a wealthy merchant carry a satchel?\nAnswer: Yes\n")

# conditioned fill: wealth -> quality (silk vs threadbare), role -> appropriate gear. Anchors the drifty slots
# (belt/satchel/nose) with explicit examples so the item stays a thing-OF-that-slot, short noun phrase. Mixed
# across wealth/role/species/kind.
FILL_FEWSHOT = (
    "A wealthy human noble. What do they wear on their torso?\nAnswer: an embroidered silk doublet\n"
    "A destitute human beggar. What do they wear on their torso?\nAnswer: a filthy threadbare tunic\n"
    "A poor goblin raider. What do they wear in their nose?\nAnswer: a crude bone nose-ring\n"
    "A modest human farmer. What do they wear on their head?\nAnswer: a wide-brimmed straw hat\n"
    "A wealthy elf merchant. What do they wear on their fingers?\nAnswer: a gold signet ring\n"
    "A poor human soldier. What do they hold in their hands?\nAnswer: a notched iron shortsword\n"
    "A modest human traveler. What do they wear on their belt?\nAnswer: a worn leather belt with a brass buckle\n"
    "A wealthy human lady. What do they carry as their satchel?\nAnswer: a beaded velvet purse\n")


def _fill_q(prefix, slot, kind):
    if kind == "held":
        return f"{prefix}. What do they hold in their {slot}?"
    if kind == "container":
        return f"{prefix}. What do they carry as their {slot}?"
    if kind in ("piercing", "jewelry"):
        return f"{prefix}. What do they wear in their {slot}?"
    return f"{prefix}. What do they wear on their {slot}?"


def has_slot(server, entity, slot, kind):
    """Presence gate — does this entity even USE this slot? Conditioned, so the poor come out sparse."""
    subj = entity_prefix(entity)
    subj = subj[0].lower() + subj[1:]
    if kind == "held":
        q = f"Is {subj} holding anything in their hands?"
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
    raw = server.gen_text(FILL_FEWSHOT + _fill_q(entity_prefix(entity), slot, kind) + "\nAnswer:",
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
