# Extraction prompt snapshots

Full rendered prompts (few-shot + a representative query) for every extraction in needs.py / traits.py. **Reference only** — nothing reads from here; regenerate with `python3 dump_prompts.py` after editing a prompt.

| file | extracts |
|---|---|
| [needs_core_applies.txt](needs_core_applies.txt) | core need sweep — does this species have need X? (yes/no, P(yes)) |
| [needs_extra.txt](needs_extra.txt) | species-specific extra needs (iter_unique, inline answer) |
| [needs_rate.txt](needs_rate.txt) | per-person rate — times/day a person satisfies a need (gen_number_median) |
| [needs_wake_hours.txt](needs_wake_hours.txt) | waking hours per day (gen_number_median) |
| [needs_affordance_gate.txt](needs_affordance_gate.txt) | does an OBJECT-KIND affect a need at all? (yes/no gate) |
| [needs_affordance_amount.txt](needs_affordance_amount.txt) | how much an object refills a need (gen_percent degree) |
| [needs_duration.txt](needs_duration.txt) | how long using an item takes (gen_duration -> minutes) |
| [needs_threshold.txt](needs_threshold.txt) | deadband — fullness at which a need demands attention (gen_number_median) |
| [needs_consumable.txt](needs_consumable.txt) | is the item used up after one use? PER SPECIES (yes/no) |
| [needs_mode_classify.txt](needs_mode_classify.txt) | need MODE — argmax of P(yes) over 5 per-mode yes/no prompts |
| [needs_provider_gate.txt](needs_provider_gate.txt) | ambient: does an item PROVIDE the condition to the area around it? (yes/no) |
| [needs_provider_amount.txt](needs_provider_amount.txt) | ambient field STRENGTH (gen_percent degree) |
| [needs_radius.txt](needs_radius.txt) | ambient field RADIUS in grid cells (gen_number_median) |
| [needs_species_affordance_gate.txt](needs_species_affordance_gate.txt) | can a CONSUMER species satisfy a need via a TARGET species? (mode-appropriate yes/no) |
| [needs_species_amount.txt](needs_species_amount.txt) | how much a target species refills the consumer's need (gen_percent) |
| [needs_species_kill.txt](needs_species_kill.txt) | does feeding kill the target? PER (consumer,target) (yes/no) |
| [traits_weight.txt](traits_weight.txt) | weight — shrink-the-unit cascade (kilograms rung shown; retries g/mg/ug) |
| [traits_size.txt](traits_size.txt) | size — shrink-the-unit cascade (centimeters rung shown; retries mm/um) |
| [traits_rarity.txt](traits_rarity.txt) | rarity — gen_percent over tier WORDS -> 0-10 (everywhere..legendary) |
| [traits_worth.txt](traits_worth.txt) | worth — gen_percent over tier WORDS -> 0-10 (worthless..priceless) |
| [traits_source.txt](traits_source.txt) | source — one category word (grown/mined/hunted/...) |
| [traits_emission_gate.txt](traits_emission_gate.txt) | emission — does the item give off light/heat/sound? (yes/no per channel) |
| [traits_emission_strength.txt](traits_emission_strength.txt) | emission STRENGTH — gen_percent over intensity WORDS (faint..blinding) |
| [traits_emission_radius.txt](traits_emission_radius.txt) | emission RADIUS in cells (gen_number_median) |
