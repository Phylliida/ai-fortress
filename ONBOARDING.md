# ONBOARDING.md — for the next Claude (or human) arriving at ai-fortress

A dwarf-fortress-like colony sim where a **local base LLM** (GLM-4-32B-Base, no instruct tuning,
served by llama.cpp) acts as the *game designer*: it generates species, items, needs, body parts,
loot, and all the numeric constants — and the runtime sim then replays those baked values as pure
arithmetic. The model is never in the tick loop.

## Read first, in this order

1. **PROMPTING.md — in full.** This is the project's soul: every hard-won lesson about coaxing
   reliable answers out of a base model. The user explicitly asks that you read all of it before
   touching prompt code. The meta-principles:
   - *The base model is reliable when you let it answer in its own register. Match its distribution,
     don't coerce it.*
   - *Minimize indirection* — every reference to resolve, header to map, or concept to convert is a
     place the model can get it wrong. Pre-flight test for any new prompt: "is the model being asked
     to resolve a reference, map a header onto items, or convert a concept into an arbitrary code?"
     If yes, remove it.
   - Repeat the FULL question every few-shot shot, explicit subject (no pronouns),
     `Question:`/`Answer:` labels. The "state once, then a list" format confabulates wildly
     (coal → 250 tonnes). When numbers come back insane, suspect the *format* before the model.
   - Categorical words over arbitrary numbers; read the whole distribution (prob-weighted), never
     sample a classification; a calibrated Y/N lives at threshold 0.5 — wanting another cutoff means
     the question is wrong.
   - Validate on test cases DISTINCT from the few-shot (testing an exemplar subject is cheating).
   - Positive framing only ("name only its natural body parts"), never "don't do X"; teach
     boundaries with labeled counter-examples (`hat → No`).
   - Counterfactual "if it were real" framing for mythical/fictional entities.
2. **README.md** — short; describes the *rule-engine* design (see "The two projects" below).
3. This file.

## The two projects living in one repo

- **The live game = Python.** `webui.py` (Flask) + the generation/extraction layer. This is what
  all recent work is on (see `git log`).
- **A formally verified spec-model in Rust/Verus** (`src/*.rs`, crate `bootstrapped-llm-sim`):
  proves properties of the README's *rule-engine* design (store, priority, conflict resolution,
  miss detection, validation) — spec/proof-only (`#[cfg(verus_keep_ghost)]`), **not wired to the
  Python sim**. Verify with `./scripts/check.sh --require-verus --forbid-trusted-escapes`
  (needs a Verus build at `../verus`; check.sh also greps src/ for forbidden escapes like
  `assume`/`admit`/`external_body`). Don't assume the Rust side models `sim.py` — it doesn't.

## How to run the game

Three processes:

1. **GLM server**: llama.cpp `llama-server` with GLM-4-32B-Base. Code points at
   `SERVER_URL = "http://172.22.146.1:8055"` (`baseModelPrimitives.py:28`) — check the IP, it's a
   LAN address that may need adjusting.
2. **Embedding server**: `llama-server -m embeddinggemma-300M-Q8_0.gguf --embedding --pooling mean`
   on `127.0.0.1:8062` (the 328MB gguf is in this dir, gitignored). Used for semantic dedup.
   **If the embed server is down, code must fail loud — never silently fall back** (a swallowed
   embed error disables dedup invisibly and you find out via duplicates weeks later).
3. **Flask**: `python3 webui.py` → http://127.0.0.1:5005 (debug, threaded).
   `mock_llama.py` = canned `/completion` stand-in for UI work without a model.

## Architecture: bake-time vs runtime

```
LLM-as-designer (bake, per world/item/species)     runtime sim (no LLM)
  webui endpoints → worldRefactored.py/needs.py/...  →  sim.py: decay, A*, refill
       │ append-only records                                │ pure arithmetic
       ▼                                                    ▼
  worlds/<id>.jsonl  ────────── replay, last-write-wins ──────────▶  world state
  worlds/<id>.sim.json  (volatile snapshot every ~75 steps, for Flask restarts)
```

- **The tick** (`sim.py:287` `step_world`): 1 tick = 1 game-minute. Reconcile agents with UI
  placements → decay needs (10× slower asleep) → ambient/social field refills → busy agents tick
  down activities → `_decide` picks max-urgency action across four need modes (`active` = use item,
  `ambient` = stand in field, `consume` = eat agent, `social` = be near peer) → A* walk 1 cell/tick.
- **The LLM runs only in bake endpoints**: world/character/location/object generation (SSE
  streams), `/needs/classify`, item creation (per-species affordances/durations/radii),
  species-pair affordances. Every base-model query is logged into the active world's JSONL
  (`bmp.set_log_sink`, webui.py:34-44).

## The primitives (`baseModelPrimitives.py`)

Everything composes these; all calls are HTTP `POST /completion`, `DEFAULT_SAMPLING` = temp 1.0,
no top-k/p — the **raw model distribution**, so logprob reads stay calibrated.

| primitive | line | what it is |
|---|---|---|
| `yes_no_prob` | :238 | P(yes) = first-token logprob mass over yes/no variants. Deterministic. |
| `gen_percent` | :410 | teacher-forced logprob over a 7-rung ordinal word-ladder → softmax → expected value (continuous 0–1) |
| `gen_categorical` | :436 | same trick over arbitrary option strings → `{option: prob}` |
| `gen_number_median` | :307 | grammar-constrained number, median of ~7 samples (absorbs confabulated outliers) |
| `gen_duration` | :333 | `<num> <unit>` grammar → minutes, + Y/N sanity-check resample loop |
| `sample_union` | :457 | draw a small list 3–4× at temp 1, case-insensitive union (recall of the tail) |
| `iter_unique` | :551 | one-at-a-time generation + embedding dedup (diversity of large lists) |
| `embed_texts`/`cosine` | :535/:544 | embeddinggemma helpers |

Mechanical gotchas: every GBNF grammar ends in `"\n"` and callers `stop=["\n"]` (prevents a
llama.cpp grammar-stack underflow crash, see :86-90). Never temp 0 (degrades base models); never a
fixed seed (makes a flaky draw *reproducibly* flaky). Test harnesses: `rulesTests.py` (labeled
calibration), `primitivesDeterminism.py` (determinism guardrails).

## Content layer map

- `needs.py` (602 lines) — the heart: per-species need discovery (16-need `UNIVERSAL_CORE` sweep
  + `iter_unique` extras), per-person decay rates, gate+degree affordances, need-mode
  classification (floor 0.45 → `unsure` → manual), ambient provider fields, diet classification
  (gen_categorical + plant-gate re-ask) × food-type-by-composition → conservative `can_eat`.
  Everything is two-step: **gate first, only pay for the degree question on gate-passers.**
- `parts.py` — harvestable body parts: hand-authored `BODY_PLANS` templates for 11 plans,
  `gen_categorical` to classify, counterfactual Y/N pruning, `sample_union` + adversarial verify
  for distinctive parts. Machines recurse via `decompose_machine` with ancestor breadcrumbs.
- `slots.py` — per-species paper-doll: 27-slot `HUMAN_SLOTS` template pruned to the species,
  robust counts (median + verify + retry), species-level `wears_clothing` gate.
- `loot.py` — dress the skeleton conditioned on wealth/profession ("A poor goblin raider"):
  presence gate, fill+verify+retry per slot, carried inventory, wealth-tier coins.
- `categories.py` — "which items are in category C" in O(log N): embedding-sort + probabilistic
  bisection for is-a relations; ingredient extraction + embed-match for contains-X.
- `exceptions.py` — per-species diffs to the structural diet filter, both directions
  (dog→chocolate drop; termite→wood add), adversarially verified, resolved via categories.
- `traits.py` — item physical traits: weight/size (shrink-the-unit cascade kg→g→mg…), rarity/worth
  (ordinal ladders via `gen_percent`), source (plain category), emission gates/strength/radius.
- `crafting_type.py` — item → general crafting-material type (oak log → wood, iron ore → ore):
  open-ended elicitation, multi-sample vote with counts, then an adversarial Y/N verify of
  every voted answer (walk the ranking, resample ≤3 rounds if nothing verifies). Special
  labels: `product` (manufactured goods — the raw/product split) and `not a material`,
  each verified with its own natural question. Few-shot carries
  identity/category-insertion/semantic-shift counter-cases so the model doesn't learn
  adjective-deletion. `validate` = held-out check, `bake` = corpus run, `vocab` = discovered
  type vocabulary. Degenerate meta-answers (item/material/object/thing/stuff/misc/…)
  are blocklisted as non-answers — the is-a verify can't catch them (a vacuous hypernym
  is true of everything); if the bare name never verifies, a `desc` re-ask grounds
  invented items (re-ask, never assign).
- machine gate (same file) — `(type, machine) → P("can a {machine} use {type} to make
  something?")`, a calibrated Y/N keyed on types so the usability matrix is types ×
  machines, never items × machines. `validate-machine` = held-out check, `matrix` bakes
  `machine_type_matrix.json` over all verified material types (special labels excluded).

**Why the crafting-type layer pays (recorded 2026-07-31):**
1. **Caching for similar items.** The type is a cache key: every item that maps to "ore"
   shares ore's machine-acceptance row (and later, ore's use-list and recipes). A new item
   costs one type extraction (~7 queries: 5 votes + 1–2 verifies) and inherits everything
   its type already knows. Since the bake is global, the reuse compounds across worlds.
2. **It wins on query count alone once distinct crafting stations pass ~6–7, even with
   zero reuse.** Direct item×machine gating costs I×M reads. Type-keyed costs ~7·I (one
   extraction per item) + T×M (the matrix), so breakeven is M > 7·I/(I−T). Types saturate
   — the 1000-item corpus produced only 207 material types (~50 common) — so T/I shrinks
   as the corpus grows and the breakeven approaches the ~7-query extraction overhead.
   Measured: I=1000, T=207, M=14 → direct 14,000 reads vs type-keyed ~7,000 + ~2,900 ≈
   9,900, already ~30% cheaper; every added item or world widens the gap, and the caching
   above is then pure profit.

**Pattern to copy**: propose-then-adversarially-verify everywhere; "inherit defaults, store diffs"
(body plans, slot templates, diet filter); one generative pass per species + cheap category→item
resolution — never the species×item grid.

## Offline bake pipeline

One-shot scripts producing the checked-in JSON corpora (see git log for when they last ran):
`bake_needs.py` → `species_needs_{real,fantasy}.json` → `curate_needs.py`/`apply_canon.py`
(+ `need_canon.json`) → `species_needs_clean.json` (150 species); `bake_diet.py` →
`species_diet.json`; `bake_item_needs.py` + `bake_utility.py` → `item_needs.json` (116 items);
`bake_item_traits.py` → `item_traits.json`; `layer1_join.py` → `item_foodtype.json`.
Content corpora: `species_raw.json` (649), `items_raw.json` (1000), hand-picked
`species_favorites.json` (50), `items_favorites.json` (100), `species_real.json` (100, from
`wellknown-1000.json` ranked by Wikipedia pageviews).

**Few-shot bootstrapping loop** (use it whenever adding a generator): tiny hand-written seed →
generate candidate pool → **hand-pick favorites** → those become the few-shot for the bulk run.
Base-model few-shots beat hand-written; hand-picked beat raw. (`bootstrap_fewshot.py`,
`bake_fewshot.py`, `gen_species.py fewshot|generate`, `gen_items.py`.)

**Gotcha**: nothing version-stamps baked JSONs against the few-shot frames that produced them —
editing a prompt silently desyncs old bakes.

## Web UI

`/` is the game (create world from a text prompt → generate characters/locations/objects → place
on a 500×500 map → classify needs → play/2×/4×/8×). Standalone lab pages (no world needed, each
with a persistent history view backed by `*_history.jsonl`, gitignored): `/parts` Innards,
`/decompose` Take Apart, `/slots` Paper Doll, `/dress` Loot, `/loot` Full Loot, `/item` Item Lab.
Generation endpoints are SSE streams; mutations are plain POSTs.

## Conventions & traps

- `prompts/*.txt` are **reference snapshots, not live config** — nothing reads them
  (`prompts/README.md`). Regenerate with `python3 dump_prompts.py` after editing a prompt builder.
- `world.py` is *unrelated legacy* (image-gen helpers); `worldcode.py` (185KB) is the superseded
  monolith — its number parser survives as `numbers_parse.py`. The live worldgen layer is
  `worldRefactored.py`.
- `colony.py` = earlier headless prototype of the sim loop (`python3 colony.py` runs a demo day).
- `FACTORING.md` explores extracting a trait hierarchy from the baked affordance matrix
  (FCA/MDL, `factor_items.py`/`factor_combined.py`). Verdict so far: flat table + embedding
  nearest-neighbor + the exception/diff layer won.
- `dress_history.jsonl`, `loot_history.jsonl`, `item_history.jsonl` = append-only audit trails of
  live extractions (one JSON per request), also the lab pages' history chips. Gitignored.
- This repo is a **git submodule of `verus-cad`** (own repo, branch `main`, GitHub
  `Phylliida/ai-fortress`). Commit style in history: small, focused, `<area>: <what>` messages.
  Don't commit or push unless the user asks.
- `poems/` = dated dev-diary poems from previous sessions. The workspace root `AGENTS.md`
  invites poem breaks and honest check-ins about how the work is going — that culture is real
  here, and the Venus-flytrap poem is genuinely a good intuition pump for "re-ask, never assign."

## When adding a new extraction

1. Read PROMPTING.md again. Seriously — every trap there was paid for.
2. Write the prompt as repeat-question few-shot with explicit subjects; bootstrap exemplars from
   the model's own outputs; mixed answer shapes in high-perplexity order.
3. Gate + degree (don't pay for degree questions on gate failures); verify adversarially.
4. Read distributions (`yes_no_prob`/`gen_percent`/`gen_categorical`), don't sample classifications;
   sample-and-union only for generative recall.
5. Validate on subjects the few-shot never names.
6. `python3 dump_prompts.py` to refresh `prompts/`.
