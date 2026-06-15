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

# ---- ordinal ladders (rarity, worth): few-shot primes the mapping, then gen_percent reads the prob of
#      EACH tier word and returns the prob-weighted expected value -> a fine-grained 0-10 score (so an item
#      the model sees as 60% rare / 40% legendary scores ~8.5, not a flat 'rare'). SCALE = [(word, 0-10)]. ----
RARITY_Q = "How rare is {s}? (everywhere, common, uncommon, rare, or legendary)"
RARITY_ANCHORS = [("water", "everywhere"), ("a loaf of bread", "common"), ("a steel sword", "uncommon"),
                  ("a diamond", "rare"), ("Excalibur", "legendary")]
RARITY_SCALE = [("everywhere", 0.0), ("common", 2.5), ("uncommon", 5.0), ("rare", 7.5), ("legendary", 10.0)]

WORTH_Q = "How much would {s} sell for? (worthless, cheap, valuable, precious, or priceless)"
WORTH_ANCHORS = [("a pebble", "worthless"), ("an apple", "cheap"), ("a steel sword", "valuable"),
                 ("a diamond", "precious"), ("the crown jewels", "priceless")]
WORTH_SCALE = [("worthless", 0.0), ("cheap", 2.5), ("valuable", 5.0), ("precious", 7.5), ("priceless", 10.0)]

# ---- categorical (source): non-ordinal, so one word off the ladder (no numeric score) ----
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


def bake_scale(server, q, anchors, scale, x):
    """Fine-grained ordinal trait (0-10): few-shot primes the mapping, then the prob-weighted expected
    value over the tier words (gen_percent reads the model's whole distribution, not just the argmax)."""
    return round(server.gen_percent(_cat_prompt(q, anchors, x), scale=scale)["value"], 2)


# ---- emission: light / heat / sound. A yes/no GATE per channel (validated: candle->light+heat, TV->all
#      three, bell->sound, coal/anvil->none), then STRENGTH (gen_percent, 0-1) and RADIUS in cells (~1m
#      each). Feeds the sim's ambient-field mechanic directly — a candle becomes a light field. ----
EMIT_GATES = {
    "light": ("Question: Does a candle give off light?\nAnswer: Yes\n"
              "Question: Does a brick give off light?\nAnswer: No\n"
              "Question: Does a lantern give off light?\nAnswer: Yes\n"
              "Question: Does a wooden chair give off light?\nAnswer: No\n"
              "Question: Does a {x} give off light?\nAnswer:"),
    "heat":  ("Question: Does a fire give off heat?\nAnswer: Yes\n"
              "Question: Does an ice cube give off heat?\nAnswer: No\n"
              "Question: Does an oven give off heat?\nAnswer: Yes\n"
              "Question: Does a stone give off heat?\nAnswer: No\n"
              "Question: Does a {x} give off heat?\nAnswer:"),
    "sound": ("Question: Does a ringing bell give off sound?\nAnswer: Yes\n"
              "Question: Does a pillow give off sound?\nAnswer: No\n"
              "Question: Does a running engine give off sound?\nAnswer: Yes\n"
              "Question: Does a rock give off sound?\nAnswer: No\n"
              "Question: Does a {x} give off sound?\nAnswer:"),
}
EMIT_RADIUS_FS = ("On a grid where each cell is about one meter:\n\n"
                  "Question: Within how many cells does the light from a campfire reach?\nAnswer: 5\n"
                  "Question: Within how many cells does the heat from a fireplace reach?\nAnswer: 3\n"
                  "Question: Within how many cells does the sound from a church bell reach?\nAnswer: 40\n"
                  "Question: Within how many cells does the {ch} from a {x} reach?\nAnswer:")

# strength as a CATEGORICAL intensity ladder per channel (the "how much…it gives off <degree>" framing
# flattened everything to ~0.65; an intensity WORD discriminates — a candle is dim, the sun is blinding).
EMIT_STRENGTH = {
    "light": ("How bright is the light from {s}? (faint, dim, bright, brilliant, or blinding)",
              [("a firefly", "faint"), ("a candle", "dim"), ("a lamp", "bright"), ("a floodlight", "brilliant"), ("the sun", "blinding")],
              [("faint", 0.1), ("dim", 0.3), ("bright", 0.55), ("brilliant", 0.8), ("blinding", 1.0)]),
    "heat":  ("How hot is {s}? (cool, warm, hot, scorching, or searing)",
              [("a cool stone", "cool"), ("a warm body", "warm"), ("a campfire", "hot"), ("an oven", "scorching"), ("molten lava", "searing")],
              [("cool", 0.1), ("warm", 0.3), ("hot", 0.55), ("scorching", 0.8), ("searing", 1.0)]),
    "sound": ("How loud is {s}? (faint, quiet, audible, loud, or deafening)",
              [("a whisper", "faint"), ("a ticking clock", "quiet"), ("a conversation", "audible"), ("a shout", "loud"), ("an explosion", "deafening")],
              [("faint", 0.1), ("quiet", 0.3), ("audible", 0.55), ("loud", 0.8), ("deafening", 1.0)]),
}


def bake_emission(server, item, gate_threshold=0.5, samples=5):
    """{channel: {strength 0-1, radius cells}} for the light/heat/sound an item gives off — channels below
    the gate are omitted. A candle's light, a fireplace's heat: drop-in for the ambient-field mechanic."""
    out = {}
    for ch, gate_fs in EMIT_GATES.items():
        if server.yes_no_prob(gate_fs.format(x=item)) < gate_threshold:
            continue
        q, anchors, scale = EMIT_STRENGTH[ch]
        strength = bake_scale(server, q, anchors, scale, item)   # categorical intensity word -> 0-1
        rr = server.gen_number_median(EMIT_RADIUS_FS.format(ch=ch, x=item), samples=samples)
        out[ch] = {"strength": round(strength, 3), "radius": max(1, round(rr["median"])) if rr else 1}
    return out


def bake_traits(server, item, samples=7):
    """The full property vector for an item subject (pass the name, or 'name: description' for ambiguous
    creative items). weight in grams, size in centimeters; rarity/worth are 0-10 scores; source is a word;
    emission is {channel: {strength, radius}}."""
    return {
        "weight_g": bake_number(server, WEIGHT_Q, WEIGHT_ANCHORS, WEIGHT_UNITS, item, samples),
        "size_cm": bake_number(server, SIZE_Q, SIZE_ANCHORS, SIZE_UNITS, item, samples),
        "rarity": bake_scale(server, RARITY_Q, RARITY_ANCHORS, RARITY_SCALE, item),
        "worth": bake_scale(server, WORTH_Q, WORTH_ANCHORS, WORTH_SCALE, item),
        "source": bake_category(server, SOURCE_Q, SOURCE_ANCHORS, SOURCE_TIERS, item),
        "emission": bake_emission(server, item),
    }
