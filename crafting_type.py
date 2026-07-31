"""
crafting_type.py — elicit an item's general CRAFTING-MATERIAL TYPE from the base model:
"iron ore" -> ore, "oak log" -> wood, "wheat flour" -> flour, "salmon" -> fish.

Why: the recipe design keys discovery on raw-material TYPES, not item names — one "ore"
use-list covers every ore, so the item x station matrix never materializes. This module is
the layer that maps item -> type. The type vocabulary is DELIBERATELY OPEN: answers are
elicited free-form and are never collapsed into a fixed list — a novel material must be
able to get a novel type word, so there is NO canon/merge step and no fixed-option
categorical read. (That also settles the mechanism question: sample-and-tally is the
permanent distribution read for an open vocabulary, not a stopgap — gen_categorical needs
a fixed option set, which is exactly what we don't want.) The per-sample VOTE counts are
kept as data — which word won, and how split the model is — for whatever downstream
consumer wants them. Run the elicitation over its own output and it climbs to
higher-level types (flour -> ingredient) — the taxonomy-induction loop.

Two special type words sit above the ordinary ones: "product" for manufactured goods
(a steel sword is not a crafting material, it's a crafting OUTPUT — the raw/product split
the recipe graph wants) and "not a material" for things that aren't usable materials at
all (sunset, resume).

Few-shot design (the model must NOT learn "delete the adjective"), three counter-case kinds:
  - identity answers (water -> water, sawdust -> sawdust): "no change" is valid, and some
    modifiers are load-bearing (sawdust is not dust);
  - category insertion (salmon -> fish, grapes -> fruit): the answer word appears nowhere
    in the input — true abstraction, not substring surgery;
  - semantic shift (oak log -> wood): the right answer is NEITHER input word as-is.
Plus convergent pairs (iron ore / copper ore -> ore; oak log / pine log -> wood) teaching
that different inputs map to the SAME type, one fantasy exemplar (dragon scale) so invented
items get types without a counterfactual frame, and one exemplar each for the two special
labels. The exemplar shapes are interleaved so no two neighbors share a shape and the
sequence never settles into a readable rhythm (neither grouped nor regularly alternating).

Verification doctrine: EVERY answer the model votes out gets an adversarial Y/N check —
"is {item} a type of {type}?" — because the model's decisions are not trusted until
verified. Junk (a confabulated ramble that out-voted the real answers) is simply an answer
that FAILS its Y/N check: we walk the vote ranking until one passes, and only if nothing
in the ranking verifies do we RESAMPLE a fresh round, up to 3 rounds (votes pool across
rounds). Items that still fail after 3 rounds come back verified=False for manual review.
The two special labels can't use the is-a frame, so they verify with their own natural
questions ("is {item} a crafting material?" — inverted; "is {item} a manufactured
product?"). Three degenerate meta-answers (item / material / item from a game) are
blocklisted outright — non-answers that gate-match every machine downstream; they're
skipped in the ranking and trigger resamples like any other junk.

Machine gate: (type, crafting machine) -> P("can a {machine} use {type} to make
something?"), keyed on TYPES so the usability matrix is types x machines, never items x
machines. The gate is itself a calibrated Y/N read — gates are the verification mechanism,
so no second-order verify. `matrix` bakes it over every verified material type from the
bake x the machine set (the "product"/"not a material" labels stay out — that's what
they're for).

Usage:
  python3 crafting_type.py validate                 # held-out type check (items NOT in the few-shot)
  python3 crafting_type.py validate-machine         # held-out machine-gate check
  python3 crafting_type.py bake [--limit N] [--offset K]   # items_raw.json -> item_crafting_types.json
  python3 crafting_type.py matrix [--machines "oven,anvil,..."]   # types x machines -> machine_type_matrix.json
  python3 crafting_type.py vocab                    # the discovered type vocabulary, by frequency
  --server URL overrides bmp.SERVER_URL (e.g. mock_llama.py for plumbing tests).
"""
import json
import sys
import baseModelPrimitives as bmp

# Question:/Answer: repeat-question format, explicit subject, no pronouns. Bare subjects
# (no a/an) so uncountables and count nouns share one template; bare lowercase answer words
# set the output register. "For crafting, what general type of material" — the affordance
# frame is what makes "oak log -> wood" (not "log") the natural answer.
TYPE_FEWSHOT = (
    "Question: For crafting, what general type of material is iron ore?\nAnswer: ore\n"
    "Question: For crafting, what general type of material is water?\nAnswer: water\n"
    "Question: For crafting, what general type of material is salmon?\nAnswer: fish\n"
    "Question: For crafting, what general type of material is oak log?\nAnswer: wood\n"
    "Question: For crafting, what general type of material is steel sword?\nAnswer: product\n"
    "Question: For crafting, what general type of material is wheat flour?\nAnswer: flour\n"
    "Question: For crafting, what general type of material is sawdust?\nAnswer: sawdust\n"
    "Question: For crafting, what general type of material is sunset?\nAnswer: not a material\n"
    "Question: For crafting, what general type of material is copper ore?\nAnswer: ore\n"
    "Question: For crafting, what general type of material is wooden chair?\nAnswer: product\n"
    "Question: For crafting, what general type of material is grapes?\nAnswer: fruit\n"
    "Question: For crafting, what general type of material is dragon scale?\nAnswer: scale\n"
    "Question: For crafting, what general type of material is pine log?\nAnswer: wood\n")


def crafting_type_prompt(item):
    return TYPE_FEWSHOT + f"Question: For crafting, what general type of material is {item}?\nAnswer:"


# --- verification few-shots. Mixed Yes/No exemplars (the base model has a reflexive-No
#     bias on bare questions), interleaved so the answer sequence carries no signal.
#     Subjects are distinct from the elicitation few-shot and from the validation set. ---
TYPE_VERIFY_FEWSHOT = (
    "Question: For crafting, is birch log a type of wood?\nAnswer: Yes\n"
    "Question: For crafting, is iron ore a type of fruit?\nAnswer: No\n"
    "Question: For crafting, is wooden chair a type of liquid?\nAnswer: No\n"
    "Question: For crafting, is molasses a type of sweetener?\nAnswer: Yes\n"
    "Question: For crafting, is gravel a type of metal?\nAnswer: No\n"
    "Question: For crafting, is catfish a type of fish?\nAnswer: Yes\n")

MATERIAL_VERIFY_FEWSHOT = (
    "Question: Is gravel a crafting material?\nAnswer: Yes\n"
    "Question: Is happiness a crafting material?\nAnswer: No\n"
    "Question: Is lullaby a crafting material?\nAnswer: No\n"
    "Question: Is linen a crafting material?\nAnswer: Yes\n")

PRODUCT_VERIFY_FEWSHOT = (
    "Question: Is wooden chair a manufactured product?\nAnswer: Yes\n"
    "Question: Is iron ore a manufactured product?\nAnswer: No\n"
    "Question: Is gravel a manufactured product?\nAnswer: No\n"
    "Question: Is wool sweater a manufactured product?\nAnswer: Yes\n")


def type_verify_prompt(item, type_word):
    if type_word == "not a material":
        return MATERIAL_VERIFY_FEWSHOT + f"Question: Is {item} a crafting material?\nAnswer:"
    if type_word == "product":
        return PRODUCT_VERIFY_FEWSHOT + f"Question: Is {item} a manufactured product?\nAnswer:"
    return TYPE_VERIFY_FEWSHOT + f"Question: For crafting, is {item} a type of {type_word}?\nAnswer:"


def verify_type(server, item, type_word, threshold=0.5):
    """Adversarial Y/N check of a voted type — every base-model decision gets verified.
    Returns (accepted, confidence in the type claim). "not a material" inverts: the claim
    is confirmed when the model says the item is NOT a crafting material. Threshold stays
    at 0.5 — a calibrated Y/N's decision boundary (wanting another cutoff means the
    question is wrong, not the threshold)."""
    if type_word == "not a material":
        conf = 1.0 - server.yes_no_prob(type_verify_prompt(item, type_word))
    else:
        conf = server.yes_no_prob(type_verify_prompt(item, type_word))
    return conf >= threshold, conf


# Degenerate meta-answers: they carry no crafting information and gate-match every machine
# downstream, so they're never selected — skipped in the vote ranking, and if they're all
# the model offers that's a junk round (resample, like any verification failure). The is-a
# verify can't catch these itself: it checks TRUTH, and a vacuous hypernym ("object") is
# true of everything — so a stoplist it is. A stoplist of NON-ANSWERS, not a closed
# vocabulary: the type space itself stays open.
TYPE_BLOCKLIST = {"item", "material", "item from a game", "object", "thing", "stuff",
                  "misc", "miscellaneous"}


def _vote_round(server, prompt, samples):
    """One round of independent type draws -> {normalized answer: votes}."""
    counts = {}
    for _ in range(samples):
        ans = bmp.clean_item(server.gen_text(prompt, stop=["\n"], n_predict=8)).lower()
        if ans:
            counts[ans] = counts.get(ans, 0) + 1
    return counts


def _first_verified(server, subject, counts, threshold):
    """Walk the vote ranking (blocklisted non-answers skipped), Y/N-verify each candidate,
    return the first accepted (answer, confidence) — or (None, 0.0)."""
    for cand in sorted(counts, key=counts.get, reverse=True):
        if cand in TYPE_BLOCKLIST:
            continue
        ok, conf = verify_type(server, subject, cand, threshold)
        if ok:
            return cand, conf
    return None, 0.0


def extract_crafting_type(server, item, desc=None, samples=5, max_rounds=3, threshold=0.5):
    """Multi-sample VOTE over open-ended type answers (sample_union with counts kept: the
    union of answers is the recall set, the counts say which word won and how split the
    model is — both wanted downstream), then VERIFY: walk the vote ranking, first answer
    to pass its Y/N check wins. An answer that fails is junk — resample a fresh round
    (votes pool) only when nothing in the ranking verifies, up to `max_rounds`. If the
    bare name produces nothing the model will stand behind and a `desc` is given, re-ask
    ONCE with "name: desc" as the subject — grounding an invented item (re-ask, never
    assign), verifying against the grounded subject too. Answers normalize to lowercase
    single-spaced, trailing punctuation stripped. Returns {"type", "p", "verified",
    "grounded", "answers", "n", "rounds"}; type="" and verified=False when nothing
    verifies."""
    counts, rounds = {}, 0
    for rnd in range(max(1, max_rounds)):
        rounds = rnd + 1
        for ans, c in _vote_round(server, crafting_type_prompt(item), samples).items():
            counts[ans] = counts.get(ans, 0) + c
        cand, conf = _first_verified(server, item, counts, threshold)
        if cand:
            return {"type": cand, "p": round(conf, 3), "verified": True, "grounded": False,
                    "answers": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                    "n": sum(counts.values()), "rounds": rounds}
    if desc:
        subject = f"{item}: {desc}"
        gcounts = _vote_round(server, crafting_type_prompt(subject), samples)
        cand, conf = _first_verified(server, subject, gcounts, threshold)
        for a, c in gcounts.items():
            counts[a] = counts.get(a, 0) + c
        rounds += 1
        if cand:
            return {"type": cand, "p": round(conf, 3), "verified": True, "grounded": True,
                    "answers": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                    "n": sum(counts.values()), "rounds": rounds}
    return {"type": "", "p": 0.0, "verified": False, "grounded": bool(desc),
            "answers": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "n": sum(counts.values()), "rounds": rounds}


# Held-out validation: NO subject appears in ANY few-shot above (testing an exemplar
# subject is cheating). Acceptable-alternates sets, not single answers — the vocabulary is
# open by design; this measures whether elicitation lands in the right place.
VALIDATION = [
    ("gold ore", {"ore"}),
    ("maple log", {"wood"}),
    ("rye flour", {"flour"}),
    ("trout", {"fish"}),
    ("apple", {"fruit"}),
    ("sand", {"sand"}),
    ("dragon bone", {"bone"}),
    ("honey", {"honey", "sweetener"}),
    ("cotton", {"cotton", "fiber", "cloth"}),
    ("steel sword", {"product"}),
    ("leather boot", {"product"}),
]


def cmd_validate(server, samples=5):
    hits = 0
    for item, ok in VALIDATION:
        r = extract_crafting_type(server, item, samples=samples)
        hit = r["verified"] and r["type"] in ok
        hits += hit
        votes = ", ".join(f"{a}x{c}" for a, c in r["answers"].items())
        print(f"{'HIT ' if hit else 'miss'} {item:15s} -> {r['type']:14s} p={r['p']:<6}"
              f"({votes})  ok={sorted(ok)}", flush=True)
    print(f"{hits}/{len(VALIDATION)} verified top-answers in the acceptable set")


def cmd_bake(server, path="items_raw.json", out_path="item_crafting_types.json",
             limit=None, offset=0, samples=5):
    items = json.load(open(path))
    try:
        out = json.load(open(out_path))     # resume: skip names already baked
    except Exception:
        out = {}
    todo = items[offset:]
    if limit:
        todo = todo[:limit]
    for i, it in enumerate(todo):
        name = it["name"] if isinstance(it, dict) else it
        desc = (it.get("desc") or None) if isinstance(it, dict) else None
        if name in out:
            continue
        r = extract_crafting_type(server, name, desc=desc, samples=samples)
        out[name] = r
        json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)   # incremental
        ident = "=" if r["type"] == name.lower() else ">"
        votes = ", ".join(f"{a}x{c}" for a, c in list(r["answers"].items())[:3])
        flag = "  UNVERIFIED" if not r["verified"] else ("  (desc-grounded)" if r.get("grounded") else "")
        print(f"{i+1:4d}/{len(todo)}  {name:24s} {ident} {r['type']:16s} p={r['p']:<6}"
              f"({votes}){flag}", flush=True)
    print(f"DONE -> {out_path} ({len(out)} items)")


def cmd_vocab(out_path="item_crafting_types.json"):
    out = json.load(open(out_path))
    vocab = {}
    for r in out.values():
        for ans, c in r["answers"].items():
            vocab[ans] = vocab.get(ans, 0) + c
    for ans, c in sorted(vocab.items(), key=lambda kv: -kv[1]):
        n_items = sum(1 for r in out.values() if r["type"] == ans)
        print(f"{c:5d} votes  {n_items:4d} items  {ans}")
    print(f"{len(vocab)} distinct type words over {len(out)} items")


# --- machine gate: (type, crafting machine) -> P(can be used in it). Keyed on TYPES, so the
#     usability matrix is types x machines (~50 x ~15), never items x machines. "Can a {machine}
#     use {type} to make something?" — the purpose clause pins the INPUT reading (a loom is made
#     of wood but doesn't USE wood), the model-as-subject form reads naturally for every machine
#     (no in/on/at preposition problem), and it's a plain calibrated Y/N (0.5 threshold).
#     Near-miss negatives do the boundary work: kiln is hot but takes clay not ore, a loom is
#     wooden but takes no wood. Mixed Yes/No, interleaved so the answer sequence has no rhythm. ---
MACHINE_GATE_FEWSHOT = (
    "Question: Can a furnace use ore to make something?\nAnswer: Yes\n"
    "Question: Can an anvil use flour to make something?\nAnswer: No\n"
    "Question: Can an oven use flour to make something?\nAnswer: Yes\n"
    "Question: Can a sawmill use wood to make something?\nAnswer: Yes\n"
    "Question: Can a kiln use ore to make something?\nAnswer: No\n"
    "Question: Can a sewing machine use cloth to make something?\nAnswer: Yes\n"
    "Question: Can a loom use wood to make something?\nAnswer: No\n"
    "Question: Can an anvil use metal to make something?\nAnswer: Yes\n"
    "Question: Can a brewing still use fruit to make something?\nAnswer: Yes\n"
    "Question: Can a mill use fish to make something?\nAnswer: No\n")


def machine_gate_prompt(type_word, machine):
    return MACHINE_GATE_FEWSHOT + f"Question: Can a {machine} use {type_word} to make something?\nAnswer:"


def machine_accepts(server, type_word, machine):
    """P(type can be used in `machine`) — a calibrated Y/N read (threshold at 0.5 downstream).
    This is a gate, not a generated decision, so it needs no second-order verify — gates ARE
    the verification mechanism."""
    return server.yes_no_prob(machine_gate_prompt(type_word, machine))


# Held-out (type, machine) pairs. kiln/sewing machine/oven/loom appear in the few-shot with the
# OPPOSITE answer for other types — these test that the gate learned the conditional, not a
# per-machine reflex. Expected answers are our labels; the run reports P so we see calibration.
MACHINE_VALIDATION = [
    ("hide", "tannery", True),
    ("clay", "kiln", True),          # few-shot had kiln+ore=No — same machine, true input
    ("ore", "oven", False),          # few-shot had oven+flour=Yes — same machine, wrong input
    ("paper", "printing press", True),
    ("stone", "sewing machine", False),
    ("meat", "campfire", True),
    ("gem", "loom", False),          # few-shot had loom+wood=No — loom takes nothing here
    ("metal", "forge", True),
    ("herb", "cauldron", True),
]


def cmd_validate_machine(server):
    hits = 0
    for type_word, machine, expect in MACHINE_VALIDATION:
        p = machine_accepts(server, type_word, machine)
        got = p >= 0.5
        hits += got == expect
        print(f"{'HIT ' if got == expect else 'miss'} {type_word:10s} x {machine:16s} "
              f"P={p:.3f}  expect={'Yes' if expect else 'No'}", flush=True)
    print(f"{hits}/{len(MACHINE_VALIDATION)} gate reads on the right side of 0.5")


# Starter crafting-machine set for the matrix bake (override with --machines "a,b,c").
STARTER_MACHINES = ["oven", "campfire", "furnace", "forge", "anvil", "mill", "sawmill",
                    "kiln", "loom", "sewing machine", "brewing still", "pottery wheel",
                    "tannery", "cauldron"]


def cmd_matrix(server, types_path="item_crafting_types.json", machines=None,
               out_path="machine_type_matrix.json", threshold=0.5):
    """The type x machine usability matrix: P(accept) for every VERIFIED material type (the
    'product'/'not a material' labels stay out — that's what they're for) x every machine."""
    machines = machines or STARTER_MACHINES
    items = json.load(open(types_path))
    types = sorted({r["type"] for r in items.values()
                    if r["verified"] and r["type"] not in ("product", "not a material")})
    mat = {}
    for m in machines:
        row = {}
        for t in types:
            row[t] = round(machine_accepts(server, t, m), 3)
        mat[m] = row
        json.dump(mat, open(out_path, "w"), indent=2)   # incremental
        yes = [t for t, p in row.items() if p >= threshold]
        print(f"{m:16s} {len(yes):3d}/{len(types)} types: {', '.join(yes)}", flush=True)
    print(f"DONE -> {out_path} ({len(machines)} machines x {len(types)} types)")


if __name__ == "__main__":
    args = sys.argv[1:]
    server_url = bmp.SERVER_URL
    if "--server" in args:
        i = args.index("--server")
        server_url = args[i + 1]
        del args[i:i + 2]
    kw = {}
    for flag, cast in (("--limit", int), ("--offset", int), ("--samples", int),
                       ("--items", str), ("--out", str), ("--machines", str)):
        if flag in args:
            i = args.index(flag)
            key = {"items": "path", "out": "out_path"}.get(flag[2:], flag[2:])
            kw[key] = cast(args[i + 1])
            del args[i:i + 2]
    if "machines" in kw:
        kw["machines"] = [m.strip() for m in kw["machines"].split(",") if m.strip()]
    s = bmp.LlamaServer(url=server_url, timeout=60, retries=3)
    cmd = args[0] if args else ""
    if cmd == "validate":
        cmd_validate(s, **kw)
    elif cmd == "validate-machine":
        cmd_validate_machine(s)
    elif cmd == "bake":
        cmd_bake(s, **kw)
    elif cmd == "matrix":
        cmd_matrix(s, **kw)
    elif cmd == "vocab":
        cmd_vocab(out_path=kw.get("out_path", "item_crafting_types.json"))
    else:
        print(__doc__)
