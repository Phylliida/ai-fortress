# Prompting GLM-4-base — lessons from the ai-fortress pipeline

Hard-won notes from building the colony sim's extraction + generation pipeline (affordances, need-modes,
item traits, species/item families) against a local **GLM-4-base** model (no instruct tuning). Builds on
Danielle's earlier few-shot rules; this collects what we learned the hard way.

## The one meta-principle

> **The base model is reliable when you let it answer in its own register. Every time we fought it, the fix
> was to *match its distribution*, not coerce it.**

Concretely, that showed up as: few-shots from the model's *own* outputs (not hand-written), **categorical
words** instead of coerced numbers, clean **adjective questions** instead of awkward "how much…<word>"
hybrids, framing that matches a **real document type it knows**, and reading its **whole distribution**
(prob-weighted) instead of just the argmax. Each is below.

## The other core lens: minimize indirection

> **Every layer of indirection is work the model can get wrong. Make the path from question to answer as
> direct as possible — nothing to resolve, map, or convert.**

This is the *why* behind several rules below; once you see it, it gives you a pre-flight test. The
indirection we kept removing, and the direct form that fixed it:

- **Pronouns** ("does *it* give off light?") → name the **explicit subject** every shot ("does a {item}…").
  A pronoun is a reference the model pays to resolve — and paying costs.
- **State-question-once + list** → **repeat the full question** each shot. The list format makes the model
  map the header question onto each row; repeating it removes that hop (and stops the confabulation).
- **Arbitrary number scales** ("rate rarity 1–10") → a **category word**. A 1–10 is a concept→arbitrary-code
  mapping the model has to invent; with a word the meaning *is* the answer. (This was the exact reason
  rarity/worth moved to words — "it has to map to its worth; just doing category words reduces the indirection.")
- **A quantity question answered with a category** ("how much would X sell for? (worthless…priceless)") →
  a **direct adjective** ("how expensive is X?"). The mismatch — asking a number, wanting a word — is itself
  indirection.

**Pre-flight test:** before sending a prompt, ask *"is the model being asked to resolve a reference, map a
header onto items, or convert a concept into an arbitrary code?"* If yes, remove it.

---

## Few-shot format (the foundation)

- **Repeat the FULL question every shot**, with an **explicit subject** (no pronouns — pronouns cost the
  model resolution), and **Question:/Answer:** labels. *This is the single biggest lever.*
- **The "state the question once, then a list" format CONFABULATES badly.** Same anchors, same model:
  - list format → `coal: 250 tonnes`, `lava lamp: 10,000,000 coins`, `Excalibur: 33 m long`
  - repeat-question format → coal 1–2 kg, sword 1 kg, diamond rare(8), Excalibur legendary(10)
  - When numbers come back insane, suspect the *format* before you suspect the model.
- Mixed Yes/No exemplars in a gate (the base model has a reflexive-"No" bias; mixed examples fix it).
- **Make the answer-shape *order* unpredictable — don't group, and don't regularly alternate either.** When
  exemplars come in shapes (Yes vs No, multi-ingredient vs single), the *sequence* of shapes should carry no
  signal, so the model is forced onto the content mapping instead of a positional one. **Both extremes leak
  and are learnable:** grouping (`YYYN` → "N is the trend now") *and* strict alternation (`YNYN` → "Y comes
  next"). Aim for a high-perplexity order where the next shape is genuinely unclear — `YYNY`, `YNYYN`.
  (Concrete miss: two multi-ingredient exemplars *then* a single one primed "the answer is a list" and raw
  items over-listed — but a strict `multi/single/multi/single` would have its own rhythm; pick an irregular
  mix.) NB: several of our gates currently use `YNYN` — fine where the content margin is huge (plant-gate
  0.96 vs 0.04), worth shuffling for borderline ones.
- For numeric anchors: **span the range** (feather 0.001 … house 100000) and **vary the subject** each shot.

## Categorical vs. numeric extraction

- **The base model is bad at calibrated numbers, good at categories.** For any trait whose unit is
  *arbitrary* (rarity, worth, intensity), extract a **word off a fixed ladder**, not a number.
- **Don't ask "how much X … <degree word>".** That hybrid flattens and conflates:
  - emission strength via "How much light does it give off? → *It gives off* ___" → candle, TV, **and the
    sun** all ~0.65 (no discrimination).
  - "How **valuable** is water?" → 6/10, because the model reads *importance-to-life*, not price.
  - Fix: a **plain adjective question** matching the tier words — `How rare is X?`, `How expensive is X?`,
    `How bright is X?` (faint…blinding). Water → cheap; sun → blinding. Clean.
- **Fine-grained ordinal score = prob-weighted expected value over the tier words** (the `gen_percent`
  trick), not argmax. Score each tier word's probability, take Σ p·value → a continuous 0–10. Then items
  land *between* tiers (lava lamp rarity 4.5, gold bar 5.1) instead of snapping to one bucket.
- **Pure categorical → read the distribution, don't sample.** `gen_categorical(prompt, options)` scores each
  option's teacher-forced sequence-logprob and softmaxes → `{option: prob}` — the categorical sibling of
  `yes_no_prob` (binary) and `gen_percent` (ordinal). The distribution's **argmax beats a single sample**
  (diet-classify *sampled* Loquat → "omnivore", a minority draw, but the distribution is sunlight 0.83), and
  you get **confidence** (Lion meat 0.99 vs Axolotl 0.56), **ambiguity-flagging** (top-prob < floor → route
  to "other"/review), and **determinism** for free. Sample only when you need a *generative* answer, not a
  classification.
- **source stays a plain category word** because it's *non-ordinal* (grown/mined/hunted/crafted/found) —
  no scale to be fine-grained over. (It's also the weakest trait: defaults to "found" on ambiguous names;
  feed the description as context if you need better.)

## Numbers that you *do* want as numbers (weight, size)

- **Shrink-the-unit cascade.** Extract in the largest unit; if the median rounds to **0**, retry in a
  smaller unit until non-zero: kg→g→mg→µg, cm→mm→µm. Strawberry → 15 g, diamond → 8 mm, pollen → 2 mg.
  - Caveat: the cascade only escalates on **0**. It can't fix a *confidently-wrong non-zero* (a bacterium
    came back "1 g" — the model has no calibration for microscopic mass). Rare for real items.
- **Unit choice matters.** *Meters* made everyday items round to 0 (a TV is 0.8 m → "0"); *centimeters*
  fixed it (TV 150 cm). Pick a unit where typical values are whole numbers ≥ 1.
- **median over ~7 samples** to absorb the wild confabulated outliers (3 was too few for wide ranges).

## Ambiguous inputs: ground from the model, re-ask don't assign

When a bare name is *ambiguous*, the fix isn't a better option-set — it's grounding the input. Lessons from
the diet-classifier (a crop name reads as the food, not the organism):

- **Open-ended FIRST to discover the categories.** Before fixing an option set, ask open-ended and read what
  the model *naturally* says. It revealed the missing category (plants → "autotroph / it doesn't eat", not
  "eats sunlight") *and* that the confusion is **name-inherent**, not the options' fault: bare "Garlic" → *"can
  eat meat and vegetables"* even with no categories given. If the open answer is wrong, fixing the options
  won't save you — fix the input.
- **Disambiguate by RE-ASKING with grounding — derived from the model, not from metadata.** Gate *"is X a
  plant?"* (a model query) → if yes, re-ask *"what does the X **plant** eat?"*. Two principles make this work:
  - *Model-derived beats a metadata field.* Deriving "is it a plant?" from a model gate (not a `kingdom`
    column) lets **exceptions surface**: a kingdom-rule forces a Venus flytrap to photosynthetic, but the
    model gate flags it as a plant *and the re-ask still answers carnivore*. (The gate matched the curated
    kingdom labels 43/43 with zero false-fires — as good as the metadata, and exception-aware.)
  - *Grounding ≠ assignment.* A re-ask clarifies the organism but lets the model's species-knowledge win —
    Venus-flytrap-*plant* → both 0.79, **not** forced to photosynthetic. *Assigning* the category from the
    gate would have hidden the exception. Re-ask, never assign.
- **Composition over taxonomy for "who can use this".** Frame compatibility by what a thing is *made of*
  ("made from animals/plants/both/neither"), not a taxonomy ("what kind of food"). It's the axis that
  actually decides compatibility, dodges the "other" dumping-ground (processed sweets → eaten by no one),
  and the "neither" bucket usefully flags inputs that aren't really the thing (a Spork "made from neither" →
  not food; it slipped a food-gate).
- **A derived filter should be CONSERVATIVE — drop only on a confident mismatch.** Its job is to remove
  *false* affordances, so keep everything ambiguous and fire only on clear cases (carnivore × pure-plant,
  herbivore × pure-animal). Over-eager filtering (an "other" food eaten by no one) reads as exceptions that
  aren't there.

## Gates and mode-appropriate questions

- **Ask the question that matches the category.** A generic gate is less reliable than a per-mode one:
  - affordance, active need: *"does **using** X satisfy the need?"*
  - affordance, ambient need: *"does X **provide** the condition to the area around it?"* — this killed the
    "a chair gives shelter" false positive (0.92 → 0.13) that the generic "relevant to" gate let through.
  - a creature as a target: *"can a {predator} satisfy hunger by **eating** a {prey}?"* vs *"…by being
    **near** a {peer}?"* — the need's *mode* picks the verb.
- **Put the conditional contrast IN the few-shot.** To learn that consumability depends on *who* uses a
  thing, the few-shot carried `termite/beam → Yes` next to `person/beam → No`. The contrast teaches the
  dependence; a single-subject few-shot can't.
- **A calibrated Y/N's threshold is 0.5 — wanting a different cutoff is a tell the *framing* is wrong.**
  `yes_no_prob` is a probability; its decision boundary is 0.5. If you reach for 0.3 to stop a gate
  mis-firing, you're papering over a question that measures the wrong predicate. Real case: pruning body
  parts with *"would {X} be a **part** of it?"* drove a lion's secretions (earwax/sweat) toward "no"
  (they're *had/produced*, not "parts"), so they sat at ~0.3–0.45 and a 0.3 cutoff "fixed" it — fragilely.
  Switching to *"would it **have** {X}?"* made them decisive (0.9+) and absences stayed 0.02, so 0.5 just
  works. Fix the question, not the cutoff. (Counterfactual *"if it were real"* belongs here too — it keeps
  a mythical creature's organs from being denied as "not real," the same reflex that once refused dragons their needs.)

## Generating lists (names + descriptions)

- **Base-model few-shots > hand-written.** In-distribution examples generalize far better. Bootstrap:
  tiny hand-written seed → generate a candidate pool → **hand-pick favorites** → use *those* as the few-shot
  for the bulk run. (Hand-written examples are off-distribution; use them only to elicit the model's own.)
- **Generate name + description together in ONE pass** as a `Name: description` line — the description is
  then coherent with the name, not bolted on by a second prompt.
- **A single generation under-samples the tail — sample a few times and UNION.** Asking once for a small list
  of alternatives drops items stochastically: one draw of mole-sauce ingredients omits *chocolate*, one
  draw of a lion's distinctive parts omits *mane*. 3–4 draws at **temp 1** + case-insensitive union recovers
  the tail at no quality cost — and the embed-match/verify downstream takes the *max* over items, so extra
  union terms don't hurt precision. Don't reach for **temp 0** (degrades the base model) or a **fixed seed**
  (only makes a flaky draw *reproducibly* flaky — it fixes determinism, not recall). `server.sample_union(
  prompt, samples=4)`. This is for **recall of a small list**; `iter_unique`'s embedding-dedup is the
  complement, for **diversity of a large one**.
- **FRAMING decides what you get — it's load-bearing.**
  - `"Monster Manual, Table of Contents:"` made GLM emit *table-of-contents meta* — page refs, "18 more
    monsters", insert-tab text → **~0/10 usable**.
  - `"A bestiary of fantasy creatures, one described per line:"` → **~9/10** real creatures.
  - For objects, **name the subtypes**: `"A catalogue … raw materials, plants, foods, tools, furnishings,
    machines, and stranger wares"` fences out animals/landscapes. ("encyclopedia" bled into geography;
    "merchant inventory" bled into livestock.)
- **The few-shot spread steers the breadth.** Pick exemplars spanning your categories (material / plant /
  food / device / cyber / magical) and the 1000 stays broad; pick five foods → a thousand foods.
- **Match the few-shot's surface style** — capitalize descriptions, and make exemplars *open differently*
  (not all "A …"), or the model copies the pattern (every line starting with "a").
- **"Slow" generation is usually a high reject rate, not slow tokens.** The ToC framing wasted ~88% of
  gens on junk; the gens themselves were 0.2–1.5 s. Check the rejection rate before blaming gen speed.

## Infrastructure

- **Embedding dedup-on-name** (cosine via embeddinggemma) gets 1000 *diverse* items from a single fixed
  prompt + warm temperature. Normalize the dedup key (`"umber Hulk"` == `"umberhulk"`) to catch spacing/case.
- **If the embed server is down, fail loud — never silently fall back.** A swallowed embed error disables
  dedup *invisibly*; you only notice via duplicates much later.
- **The Wikipedia URL slug is the common name** (`Orcinus orca` → `/wiki/Orca`) — handy when curating real
  taxa: slug→name, drop binomial-only slugs (obscure long-tail), dedup, rank by pageviews.

## Quick symptom → fix

| symptom | fix |
|---|---|
| numbers wildly wrong (250-tonne coal) | repeat the full question every shot (not list format) |
| an ordinal score flat (~0.65 everywhere) | categorical words + gen_percent expected value, not "how much…<word>" |
| "valuable" means *important* not *priced* | switch to a price adjective ("how expensive") |
| tiny item → 0 (weight/size) | shrink-the-unit cascade; pick a unit where values are ≥ 1 |
| generation slow | measure the reject rate — it's usually framing producing junk, not gen speed |
| generator emits meta / off-topic | reframe to a real document type, name the subtypes |
| a generated short list keeps dropping items | sample 3–4× at temp 1 and union (`sample_union`), not one draw |
| diversity collapses (1000 near-dups) | embedding dedup-on-name; diversify the few-shot exemplars |
| a gate over-fires (chair→shelter) | ask the *mode-appropriate* question, not a generic one |
| you want a non-0.5 Y/N threshold (e.g. 0.3) | the framing measures the wrong predicate — fix the question, keep 0.5 |
| a gate denies a fictional thing its parts/needs | counterfactual framing: "if a {X} were real, would it…" |
| categorical argmax flaky / want confidence | `gen_categorical` — read the distribution, don't sample |
| bare name read as the wrong sense (garlic→food) | model-gate "is it a plant?" → re-ask "X plant" (don't assign from metadata) |
| a structural rule hides real exceptions | derive the rule from a model query, and re-ask — never assign from a metadata field |
| a derived filter flags exceptions that aren't there | make it conservative — drop only confident mismatches, keep ambiguous |
