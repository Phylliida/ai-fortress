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
    by the wording the wealth tier elicits ('tattered leather' vs 'fine wool')."""
    raw = server.gen_text(FILL_FEWSHOT + _fill_q(entity_prefix(entity), slot, kind) + "\nAnswer:",
                          stop=["\n"], n_predict=20)
    item = raw.strip().strip(".").strip()
    return None if not item or item.lower() in parts.NULL_TOKENS else item
