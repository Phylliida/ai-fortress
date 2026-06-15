"""
traits.py — extract per-ITEM property traits with GLM (the item-property analog of needs.py's need bakes).

Every trait is a HAND-ANCHORED, REPEAT-QUESTION few-shot (the full question restated each shot, explicit
subject, no pronouns, Question:/Answer: labels — the format that makes base-model extraction reliable; the
'state the question once + list' format confabulates). Two kinds:

  NUMERIC (weight, size) — a SHRINK-THE-UNIT cascade: extract in the largest unit, and if the median rounds
    to 0, retry in a smaller unit until non-zero. So a strawberry comes back 15 g (not 0 kg), a diamond 8 mm.
    Anchors are written once in a BASE unit and re-rendered per rung, and the result is returned in the base
    unit (grams / centimeters) so every item is one comparable number.

  CATEGORICAL (rarity, worth, source) — a single word off a fixed ladder. Their "units" are arbitrary, so the
    WORD is the meaning (no number indirection); snapped to the allowed set if the model drifts.
"""

# ---- numeric cascades: (question, anchors-in-base-unit, [(unit_name, base_units_per_unit) big->small]) ----
WEIGHT_Q = "About how much does {s} weigh, in {u}?"
WEIGHT_ANCHORS = [("a feather", 1), ("an apple", 200), ("a brick", 2000), ("a person", 70000), ("a car", 1500000)]  # grams
WEIGHT_UNITS = [("kilograms", 1000), ("grams", 1), ("milligrams", 1e-3), ("micrograms", 1e-6)]   # grams per unit
WEIGHT_BASE = "grams"

SIZE_Q = "About how large is {s} across its longest dimension, in {u}?"
SIZE_ANCHORS = [("a grain of rice", 0.5), ("a coin", 2), ("a book", 25), ("a chair", 100), ("a car", 400), ("a house", 1200)]  # cm
SIZE_UNITS = [("centimeters", 1), ("millimeters", 0.1), ("micrometers", 1e-4)]   # cm per unit
SIZE_BASE = "centimeters"

# ---- categorical ladders: (question, [(subject, tier)], allowed-tiers low->high) ----
RARITY_Q = "How rare is {s}? (everywhere, common, uncommon, rare, or legendary)"
RARITY_ANCHORS = [("water", "everywhere"), ("a loaf of bread", "common"), ("a steel sword", "uncommon"),
                  ("a diamond", "rare"), ("Excalibur", "legendary")]
RARITY_TIERS = ["everywhere", "common", "uncommon", "rare", "legendary"]

WORTH_Q = "How valuable is {s}? (worthless, cheap, valuable, precious, or priceless)"
WORTH_ANCHORS = [("a pebble", "worthless"), ("an apple", "cheap"), ("a steel sword", "valuable"),
                 ("a diamond", "precious"), ("the crown jewels", "priceless")]
WORTH_TIERS = ["worthless", "cheap", "valuable", "precious", "priceless"]

SOURCE_Q = "How is {s} obtained? (grown, foraged, mined, hunted, crafted, manufactured, or found)"
SOURCE_ANCHORS = [("wheat", "grown"), ("a wild mushroom", "foraged"), ("iron ore", "mined"),
                  ("venison", "hunted"), ("a wooden chair", "crafted"), ("a plastic bottle", "manufactured"),
                  ("a seashell", "found")]
SOURCE_TIERS = ["grown", "foraged", "mined", "hunted", "crafted", "manufactured", "found"]


def _cascade_prompt(q, anchors, unit, base_per_unit, x):
    shots = "".join(f"Question: {q.format(s=a, u=unit)}\nAnswer: {v / base_per_unit:g}\n" for a, v in anchors)
    return shots + f"Question: {q.format(s=x, u=unit)}\nAnswer:"


def bake_number(server, q, anchors, units, x, samples=7):
    """Numeric trait via the shrink-the-unit cascade. Returns the value in BASE units (the unit with
    factor 1), so items are comparable; 0.0 only if it's zero even at the smallest rung."""
    for unit, bpu in units:
        m = server.gen_number_median(_cascade_prompt(q, anchors, unit, bpu, x), samples=samples)["median"]
        if m > 0:
            return round(m * bpu, 6)
    return 0.0


def _cat_prompt(q, anchors, x):
    return "".join(f"Question: {q.format(s=a)}\nAnswer: {v}\n" for a, v in anchors) + f"Question: {q.format(s=x)}\nAnswer:"


def bake_category(server, q, anchors, tiers, x):
    """One category word via repeat-question few-shot, snapped to the allowed ladder if the model drifts."""
    raw = server.gen_text(_cat_prompt(q, anchors, x), stop=["\n"], n_predict=6).strip().lower().rstrip(".")
    for t in tiers:
        if t in raw:
            return t
    return raw or tiers[0]


def bake_traits(server, item, samples=7):
    """The full property vector for an item subject (pass the name, or 'name: description' for ambiguous
    creative items). weight in grams, size in centimeters; rarity/worth/source are words off their ladders."""
    return {
        "weight_g": bake_number(server, WEIGHT_Q, WEIGHT_ANCHORS, WEIGHT_UNITS, item, samples),
        "size_cm": bake_number(server, SIZE_Q, SIZE_ANCHORS, SIZE_UNITS, item, samples),
        "rarity": bake_category(server, RARITY_Q, RARITY_ANCHORS, RARITY_TIERS, item),
        "worth": bake_category(server, WORTH_Q, WORTH_ANCHORS, WORTH_TIERS, item),
        "source": bake_category(server, SOURCE_Q, SOURCE_ANCHORS, SOURCE_TIERS, item),
    }
