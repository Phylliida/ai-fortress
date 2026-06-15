"""
dump_prompts.py — render every extraction prompt (few-shot + a representative query) to prompts/*.txt
for inspection. REFERENCE ONLY — nothing reads from prompts/; the live prompts are built in needs.py /
traits.py. Re-run this after editing a prompt to refresh the snapshot.
"""
import os, json
import needs as N
import traits as T

os.makedirs("prompts", exist_ok=True)
fav = {s["name"]: s for s in json.load(open("species_favorites.json"))}
makit = fav["Makit"]["desc"]

# each: (filename, one-line description, example-args note, rendered prompt text)
items = []

def add(fn, desc, note, text):
    items.append((fn, desc, note, text))

# ---------------- needs.py ----------------
add("needs_core_applies.txt", "core need sweep — does this species have need X? (yes/no, P(yes))",
    "real: ('Tomato plant','water')   |   fantasy: ('Makit','food', desc=...)",
    "### REAL (no description) ###\n" + N.need_applies_prompt("Tomato plant", "water") +
    "\n\n\n### FANTASY (description grounds the invented name) ###\n" + N.need_applies_prompt("Makit", "food", desc=makit))
add("needs_extra.txt", "species-specific extra needs (iter_unique, inline answer)",
    "('Tiger', have=['food','water','sleep'])",
    N.extra_needs_prompt("Tiger", ["food", "water", "sleep"]))
add("needs_rate.txt", "per-person rate — times/day a person satisfies a need (gen_number_median)",
    "('a hardworking baker', 'social')", N.rate_prompt("a hardworking baker", "social"))
add("needs_wake_hours.txt", "waking hours per day (gen_number_median)",
    "('A hardworking baker who rises before dawn.')",
    N.wake_hours_prompt("A hardworking baker who rises before dawn."))
add("needs_affordance_gate.txt", "does an OBJECT-KIND affect a need at all? (yes/no gate)",
    "('person','fireplace','warmth')", N.affordance_applies_prompt("person", "fireplace", "warmth"))
add("needs_affordance_amount.txt", "how much an object refills a need (gen_percent degree)",
    "('person','hot meal','food')", N.affordance_amount_prompt("person", "hot meal", "food"))
add("needs_duration.txt", "how long using an item takes (gen_duration -> minutes)",
    "('person','bed','sleep')", N.duration_prompt("person", "bed", "sleep"))
add("needs_threshold.txt", "deadband — fullness at which a need demands attention (gen_number_median)",
    "('hygiene')", N.threshold_prompt("hygiene"))
add("needs_consumable.txt", "is the item used up after one use? PER SPECIES (yes/no)",
    "('termite','wooden beam')", N.consumable_prompt("termite", "wooden beam"))
add("needs_mode_classify.txt", "need MODE — argmax of P(yes) over 5 per-mode yes/no prompts",
    "need='hunger', all 5 modes shown",
    "\n\n\n".join(f"### MODE: {m} ###\n" + N.mode_prompt(m, "hunger") for m in N.NEED_MODES))
add("needs_provider_gate.txt", "ambient: does an item PROVIDE the condition to the area around it? (yes/no)",
    "('campfire','warmth')", N.provider_applies_prompt("campfire", "warmth"))
add("needs_provider_amount.txt", "ambient field STRENGTH (gen_percent degree)",
    "('campfire','warmth')", N.provider_amount_prompt("campfire", "warmth"))
add("needs_radius.txt", "ambient field RADIUS in grid cells (gen_number_median)",
    "('watchtower','safety')", N.radius_prompt("watchtower", "safety"))
add("needs_species_affordance_gate.txt", "can a CONSUMER species satisfy a need via a TARGET species? (mode-appropriate yes/no)",
    "consume: ('wolf','rabbit','food')   |   social: ('person','dog','social')",
    "### CONSUME mode (eat the target) ###\n" + N.species_affordance_prompt("consume", "wolf", "rabbit", "food") +
    "\n\n\n### SOCIAL/AMBIENT mode (be near the target) ###\n" + N.species_affordance_prompt("social", "person", "dog", "social"))
add("needs_species_amount.txt", "how much a target species refills the consumer's need (gen_percent)",
    "('consume','wolf','rabbit','food')", N.species_amount_prompt("consume", "wolf", "rabbit", "food"))
add("needs_species_kill.txt", "does feeding kill the target? PER (consumer,target) (yes/no)",
    "('wolf','rabbit')", N.species_kill_prompt("wolf", "rabbit"))

# ---------------- traits.py ----------------
add("traits_weight.txt", "weight — shrink-the-unit cascade (kilograms rung shown; retries g/mg/ug)",
    "('a diamond')  unit=kilograms",
    T._cascade_prompt(T.WEIGHT_Q, T.WEIGHT_ANCHORS, "kilograms", 1000, "a diamond"))
add("traits_size.txt", "size — shrink-the-unit cascade (centimeters rung shown; retries mm/um)",
    "('a diamond')  unit=centimeters",
    T._cascade_prompt(T.SIZE_Q, T.SIZE_ANCHORS, "centimeters", 1, "a diamond"))
add("traits_rarity.txt", "rarity — gen_percent over tier WORDS -> 0-10 (everywhere..legendary)",
    "('a diamond')", T._cat_prompt(T.RARITY_Q, T.RARITY_ANCHORS, "a diamond"))
add("traits_worth.txt", "worth — gen_percent over tier WORDS -> 0-10 (worthless..priceless)",
    "('a diamond')", T._cat_prompt(T.WORTH_Q, T.WORTH_ANCHORS, "a diamond"))
add("traits_source.txt", "source — one category word (grown/mined/hunted/...)",
    "('a diamond')", T._cat_prompt(T.SOURCE_Q, T.SOURCE_ANCHORS, "a diamond"))
add("traits_emission_gate.txt", "emission — does the item give off light/heat/sound? (yes/no per channel)",
    "light channel, ('a candle')", T.EMIT_GATES["light"].format(x="a candle"))
add("traits_emission_strength.txt", "emission STRENGTH — gen_percent over intensity WORDS (faint..blinding)",
    "light channel, ('a candle')",
    T._cat_prompt(T.EMIT_STRENGTH["light"][0], T.EMIT_STRENGTH["light"][1], "a candle"))
add("traits_emission_radius.txt", "emission RADIUS in cells (gen_number_median)",
    "light channel, ('a candle')", T.EMIT_RADIUS_FS.format(ch="light", x="a candle"))

for fn, desc, note, text in items:
    with open(f"prompts/{fn}", "w") as f:
        f.write(f"# {desc}\n# example args: {note}\n# (reference snapshot — regenerate with dump_prompts.py)\n"
                + "#" + "=" * 78 + "\n\n" + text + ("\n" if not text.endswith("\n") else ""))

# index
with open("prompts/README.md", "w") as f:
    f.write("# Extraction prompt snapshots\n\nFull rendered prompts (few-shot + a representative query) for "
            "every extraction in needs.py / traits.py. **Reference only** — nothing reads from here; "
            "regenerate with `python3 dump_prompts.py` after editing a prompt.\n\n")
    f.write("| file | extracts |\n|---|---|\n")
    for fn, desc, note, _ in items:
        f.write(f"| [{fn}]({fn}) | {desc} |\n")

print(f"wrote {len(items)} prompt files + README to prompts/")
for fn, *_ in items:
    print("  ", fn)
