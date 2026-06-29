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


# conditioned fill: wealth -> quality (silk vs threadbare), role -> appropriate gear, and "nothing" for empty
# slots. Mixed across wealth/role/species, includes an empty (goblin fingers) and a held weapon.
FILL_FEWSHOT = (
    "A wealthy human noble. What do they wear on their torso?\nAnswer: an embroidered silk doublet\n"
    "A destitute human beggar. What do they wear on their torso?\nAnswer: a filthy threadbare tunic\n"
    "A poor goblin raider. What do they wear on their fingers?\nAnswer: nothing\n"
    "A modest human farmer. What do they wear on their head?\nAnswer: a wide-brimmed straw hat\n"
    "A wealthy elf merchant. What do they wear on their fingers?\nAnswer: a gold signet ring\n"
    "A poor human soldier. What do they hold in their hands?\nAnswer: a notched iron shortsword\n")


def _fill_q(prefix, slot, kind):
    if kind == "held":
        return f"{prefix}. What do they hold in their {slot}?"
    if kind in ("piercing", "jewelry"):
        return f"{prefix}. What do they wear in their {slot}?"
    return f"{prefix}. What do they wear on their {slot}?"


def fill_slot(server, entity, slot, kind="worn"):
    """The item in `slot` for this entity (or None if empty), conditioned on who they are. Quality is carried
    by the wording the wealth tier elicits ('tattered leather' vs 'fine wool'). The phrase naturally captures
    plurality where it applies ('silver rings', 'a sword and shield'), so multi-capacity slots need no special
    case — the skeleton's count is the body capacity, the phrase is what's actually worn."""
    raw = server.gen_text(FILL_FEWSHOT + _fill_q(entity_prefix(entity), slot, kind) + "\nAnswer:",
                          stop=["\n"], n_predict=20)
    item = raw.strip().strip(".").strip()
    return None if not item or item.lower() in parts.NULL_TOKENS else item


def fill_worn(server, entity, skeleton):
    """Dress every slot in the species `skeleton` (from slots.species_slots), conditioned on the entity.
    Returns the FILLED slots as [{slot, kind, count, item}] — empty slots are dropped."""
    out = []
    for s in skeleton:
        item = fill_slot(server, entity, s["slot"], s["kind"])
        if item:
            out.append({"slot": s["slot"], "kind": s["kind"], "count": s["count"], "item": item})
    return out


def dress(server, entity, desc=None):
    """Full head-to-toe: build the species paper-doll, then fill it for this entity."""
    import slots
    skeleton = slots.species_slots(server, entity["species"], desc)
    return fill_worn(server, entity, skeleton)
