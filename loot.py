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
    "Question: If a destitute beggar were real, would they wear something on their torso?\nAnswer: Yes\n"   # basic: even rags
    "Question: If a poor goblin raider were real, would they wear a nose ring?\nAnswer: No\n"               # luxury: wealth-gated
    "Question: If a destitute beggar were real, would they wear rings on their fingers?\nAnswer: No\n"
    "Question: If a poor goblin raider were real, would they carry a weapon or tool?\nAnswer: Yes\n"        # combatant: armed
    "Question: If a wealthy merchant were real, would they carry a satchel?\nAnswer: Yes\n"
    "Question: If a poor farmer were real, would they wear a gold necklace?\nAnswer: No\n")

# conditioned fill: wealth -> quality (silk vs threadbare), role -> appropriate gear. Anchors the drifty slots
# (belt/satchel/nose) with explicit examples so the item stays a thing-OF-that-slot, short noun phrase. Per the
# guidelines: EXPLICIT SUBJECT in the question (no 'they' pronoun), Question:/Answer: labels, mixed wealth/role/
# species/kind.
# answers OMIT the leading article (a/an/the) — saves a generation token per item and reads cleaner in a list.
# COUNTERFACTUAL "if … were real, what would they" so mythical entities (a wraith, a lich) don't balk at being
# dressed (same trick as the slot prune that kept dragons their organs).
FILL_FEWSHOT = (
    "Question: If a wealthy human noble were real, what would they wear on their torso?\nAnswer: embroidered silk doublet\n"
    "Question: If a destitute human beggar were real, what would they wear on their torso?\nAnswer: filthy threadbare tunic\n"
    "Question: If a poor goblin raider were real, what would they wear in their nose?\nAnswer: crude bone nose-ring\n"
    "Question: If a modest human farmer were real, what would they wear on their head?\nAnswer: wide-brimmed straw hat\n"
    "Question: If a wealthy elf merchant were real, what would they wear on their fingers?\nAnswer: gold signet ring\n"
    "Question: If a poor human soldier were real, what would they hold in their hands?\nAnswer: notched iron shortsword\n"
    "Question: If a modest human traveler were real, what would they wear on their belt?\nAnswer: worn leather belt with a brass buckle\n"
    "Question: If a wealthy human lady were real, what would they carry as their satchel?\nAnswer: beaded velvet purse\n")

# slot-adherence verify: is the generated item actually a thing-OF-that-slot? Catches the drift (belt -> "oak
# canteen") and garbles. Two question forms (a-kind-of / holdable) matched to the slot kind. Mixed Y/N,
# high-perplexity order (Y N N Y Y).
FITS_FEWSHOT = (
    "Question: Is a silk doublet a kind of shirt?\nAnswer: Yes\n"
    "Question: Is a carved oak canteen a kind of belt?\nAnswer: No\n"
    "Question: Could a wide-brimmed straw hat be held in the hands?\nAnswer: No\n"
    "Question: Is a silver hoop a kind of earring?\nAnswer: Yes\n"
    "Question: Could a notched iron sword be held in the hands?\nAnswer: Yes\n")


def _subj(entity):
    p = entity_prefix(entity)
    return p[0].lower() + p[1:]                          # 'a poor goblin raider' for in-question use


def _fill_q(subj, slot, kind):
    if kind == "held":
        verb = "hold in their hands"
    elif kind == "container":
        verb = f"carry as their {slot}"
    elif kind in ("piercing", "jewelry"):
        verb = f"wear in their {slot}"
    else:
        verb = f"wear on their {slot}"
    return f"If {subj} were real, what would they {verb}?"


def has_slot(server, entity, slot, kind):
    """Presence gate — does this entity even USE this slot? Conditioned, so luxuries are wealth-gated but
    basics survive for the poor."""
    subj = _subj(entity)
    if kind == "held":
        verb = "carry a weapon or tool"
    elif kind == "container":
        verb = f"carry a {slot}"
    elif kind in ("piercing", "jewelry"):
        verb = f"wear a {slot}"
    else:
        verb = f"wear something on their {slot}"
    return server.yes_no_prob(PRESENCE_FEWSHOT + f"Question: If {subj} were real, would they {verb}?\nAnswer:") >= 0.5


def _strip_article(s):
    """Drop a leading 'a '/'an '/'the ' (backstop — the few-shot already omits it)."""
    low = s.lower()
    for art in ("an ", "a ", "the "):
        if low.startswith(art):
            return s[len(art):]
    return s


def fits_slot(server, item, slot, kind):
    """Slot-adherence: is `item` actually a thing that belongs in this slot? (a-kind-of for worn/jewelry/
    container, holdable for held). Rejects belt->'oak canteen', shirt->'a pendant', garbles."""
    q = f"Could {item} be held in the hands?" if kind == "held" else f"Is {item} a kind of {slot}?"
    return server.yes_no_prob(FITS_FEWSHOT + f"Question: {q}\nAnswer:") >= 0.5


def fill_slot(server, entity, slot, kind="worn", max_words=7, tries=3):
    """The item in `slot` for this entity (or None), conditioned on who they are. Generate -> guard (length/
    null) -> slot-adherence verify; retry up to `tries` (sampling gives variety), else None (drop the slot
    rather than show a mismatched item). Quality rides the wealth wording; plurality rides the phrase."""
    prompt = FILL_FEWSHOT + "Question: " + _fill_q(_subj(entity), slot, kind) + "\nAnswer:"
    for _ in range(tries):
        item = _strip_article(server.gen_text(prompt, stop=["\n", "."], n_predict=14).strip().strip(".,").strip())
        if not item or item.lower() in parts.NULL_TOKENS or len(item.split()) > max_words:
            continue
        if fits_slot(server, item, slot, kind):
            return item
    return None


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


# ---------- carried inventory (the bag): category-conditioned lists + coins as a wealth-tier amount ----------
# (key, generation noun, verify TYPE) — the type drives the category-adherence check ("is X a weapon?"), which
# both keeps items on-category (no salt-shaker under weapons) and drops garbles (a "tiny" isn't a weapon either).
INVENTORY_CATEGORIES = [
    ("weapons",     "spare weapons",           "a weapon"),
    ("tools",       "tools or useful gear",    "a tool or piece of gear"),
    ("consumables", "food, drink, or potions", "something to eat, drink, or use up"),
    ("valuables",   "valuables or trinkets",   "a valuable or trinket"),
    ("personal",    "personal belongings",     "a personal belonging"),
]

# one exemplar per category — explicit subject, Question:/Answer:, multi-answer VARYING counts (2/3/3/1/2),
# articles dropped, wealth/role spread. Empty categories fall out naturally (a jeweler carries no spare weapons).
CARRY_FEWSHOT = (
    "Question: If a poor goblin raider were real, what spare weapons would they carry?\nAnswer: throwing knives, rusty shiv\n"
    "Question: If a wealthy elf merchant were real, what tools or useful gear would they carry?\nAnswer: brass scale, abacus, wax seal\n"
    "Question: If a poor human soldier were real, what food, drink, or potions would they carry?\nAnswer: hardtack, dried meat, waterskin\n"
    "Question: If a destitute beggar were real, what valuables or trinkets would they carry?\nAnswer: chipped glass bead\n"
    "Question: If a modest human farmer were real, what personal belongings would they carry?\nAnswer: wooden charm, folded letter\n")
# plausibility verify: would THIS entity actually carry it? (a beggar wouldn't have a crown). Y N N Y.
CARRY_VERIFY_FEWSHOT = (
    "Question: If a poor goblin raider were real, would they carry a rat skull?\nAnswer: Yes\n"
    "Question: If a destitute beggar were real, would they carry a golden crown?\nAnswer: No\n"
    "Question: If a peasant farmer were real, would they carry a royal scepter?\nAnswer: No\n"
    "Question: If a wealthy merchant were real, would they carry a coin purse?\nAnswer: Yes\n")
# category-adherence verify: is the item actually that TYPE of thing? Keeps categories clean (salt-shaker is
# not a weapon) and drops garbles ('tiny' is not a tool). Mixed types, high-perplexity order (Y N N Y Y).
CATEGORY_FITS_FEWSHOT = (
    "Question: Is a rusty dagger a weapon?\nAnswer: Yes\n"
    "Question: Is a salt shaker a weapon?\nAnswer: No\n"
    "Question: Is bare hands a tool or piece of gear?\nAnswer: No\n"
    "Question: Is dried meat something to eat, drink, or use up?\nAnswer: Yes\n"
    "Question: Is a magnifying glass a tool or piece of gear?\nAnswer: Yes\n")
# coins as a CATEGORICAL amount (categories beat numbers for base models), wealth-graded. Keeps its article.
COINS_FEWSHOT = (
    "Question: If a destitute beggar were real, how much money would they carry?\nAnswer: a few copper coins\n"
    "Question: If a wealthy merchant were real, how much money would they carry?\nAnswer: a heavy purse of gold\n"
    "Question: If a modest farmer were real, how much money would they carry?\nAnswer: a handful of silver\n"
    "Question: If a poor soldier were real, how much money would they carry?\nAnswer: a small pouch of copper\n")


def category_fits(server, item, type_noun):
    """Category-adherence: is `item` actually that TYPE (a weapon / a tool / …)? Rejects mis-filed items and
    garbles (a non-thing fails every type)."""
    return server.yes_no_prob(CATEGORY_FITS_FEWSHOT + f"Question: Is {item} {type_noun}?\nAnswer:") >= 0.5


def carry_category(server, entity, noun, type_noun, samples=4, threshold=0.5, max_words=4):
    """Items in one carry category for this entity: sample-union (recall) -> embed-dedup -> TWO verifies:
    category-adherence (is it actually a {type}? — also kills garbles) AND plausibility (would they carry it?).
    Articles stripped. Empty list when they'd carry nothing of the kind."""
    subj = _subj(entity)
    raw = server.sample_union(CARRY_FEWSHOT + f"Question: If {subj} were real, what {noun} would they carry?\nAnswer:",
                              samples=samples, n_predict=40, max_words=max_words, reject=lambda k: k in parts.NULL_TOKENS)
    out = []
    for it in parts.embed_dedup([_strip_article(x) for x in raw]):
        if (category_fits(server, it, type_noun) and
                server.yes_no_prob(CARRY_VERIFY_FEWSHOT + f"Question: If {subj} were real, would they carry {it}?\nAnswer:") >= threshold):
            out.append(it)
    return out


def carry_coins(server, entity):
    """A wealth-graded coin amount as a descriptive phrase ('a heavy purse of gold' / 'a few copper coins')."""
    subj = _subj(entity)
    raw = server.gen_text(COINS_FEWSHOT + f"Question: If {subj} were real, how much money would they carry?\nAnswer:",
                          stop=["\n", "."], n_predict=12)
    return raw.strip().strip(".,").strip() or None


def carry_inventory(server, entity):
    """The whole bag: coins (wealth-tier amount) + items per category."""
    return {"coins": carry_coins(server, entity),
            "categories": {cat: carry_category(server, entity, noun, typ) for cat, noun, typ in INVENTORY_CATEGORIES}}
