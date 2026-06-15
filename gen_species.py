#!/usr/bin/env python3
"""
gen_species.py — generate a fantasy-species list (name + description together) with GLM, in two steps.

Each entry is generated in ONE pass as a Monster-Manual line "Name: description", so the description is
coherent with the name instead of bolted on after. Two human-in-the-loop steps, because base-model
few-shots beat hand-written ones AND hand-picked ones beat raw ones:

  STEP 1 — candidates from a tiny HAND-WRITTEN seed (bootstrap only):
      python3 gen_species.py fewshot [N=150] [out=fewshot_candidates.json]
    Generate N entries with the hand-written SEED_ENTRIES few-shot. Pick your top ~3 favourites into
    species_fewshot.json — those base-model entries REPLACE the hand-written ones.

  STEP 2 — the polished list, few-shotting from your hand-picked entries:
      python3 gen_species.py generate [N=1000] [fewshot=species_fewshot.json] [out=species_raw.json]
    Curate the result into your final species list.

Resamples one entry at a time (warm temp), dedups on the NAME with embeddinggemma, writes incrementally
(a kill won't lose progress), and is resumable (re-running tops up to N).
"""
import json
import os
import re
import sys
import baseModelPrimitives as bmp

# Hand-written bootstrap entries — used ONLY to seed STEP 1, then replaced by your base-model-generated
# picks. (Off-distribution hand-written examples are exactly what we want to get OUT of the few-shot.)
SEED_ENTRIES = [   # capitalized, and each opens DIFFERENTLY so the model doesn't start every line with "A"
    ("Dragon", "Colossal winged reptiles that hoard gold and breathe gouts of fire."),
    ("Goblin", "Green-skinned humanoids, cunning but cowardly, that raid farms in chittering packs."),
    ("Treant", "Guardians of the oldest forests — slow, towering tree-folk that wake once a century."),
]


def manual_prompt(entries):
    """A Monster-Manual table of contents the model continues; each line is 'Name: description', so one
    completion yields a NAME and a matching DESCRIPTION together."""
    lines = "".join(f"- {n[:1].upper() + n[1:]}: {d}\n" for n, d in entries)
    return "Monster Manual, Table of Contents:\n" + lines + "- "


def parse_entry(raw):
    """'Dragonkin: a draconic warrior...' -> ('Dragonkin', 'a draconic warrior...'); None if malformed."""
    if ":" not in raw:
        return None
    name, desc = raw.split(":", 1)
    name, desc = name.strip(), desc.strip()
    desc = desc[:1].upper() + desc[1:]                     # capitalize the description (belt-and-suspenders)
    return (name, desc) if name and desc else None


def _reject(name):
    """Drop obvious degenerate names (the curator shouldn't have to)."""
    return (bool(re.search(r'[:;,\d"*|()_]', name))        # punctuation/markdown that signals garbled output
            or len(name.split()) > 3                        # real species names are short
            or any(ord(c) > 0x024F for c in name))          # non-Latin script (GLM occasionally drifts)


def _key(name):
    return re.sub(r"\s+", "", name.lower())               # collapse case/spacing: 'umber Hulk' == 'umberhulk'


def generate(server, fewshot, n, existing, sim_threshold=0.78):
    """Yield (name, desc): resample the manual prompt, parse one entry, dedup on the NAME (embeddinggemma)."""
    embs, seen = [], set()
    for name, _ in existing:                               # seed dedup with what we already have
        seen.add(_key(name))
        try:
            embs.append(bmp.embed_texts([name])[0])
        except Exception:
            pass
    prompt = manual_prompt(fewshot)
    got, attempts = 0, 0
    while got < n and attempts < max(50, n * 12):
        attempts += 1
        parsed = parse_entry(server.gen_text(prompt, stop=["\n"], n_predict=70))
        if not parsed:
            continue
        name, desc = parsed
        if _reject(name) or _key(name) in seen:
            continue
        try:
            e = bmp.embed_texts([name])[0]
            if embs and max(bmp.cosine(e, pe) for pe in embs) >= sim_threshold:
                continue
        except Exception:
            e = None
        seen.add(_key(name))
        if e is not None:
            embs.append(e)
        got += 1
        yield name, desc


def run(fewshot, n, out):
    server = bmp.LlamaServer(timeout=60, retries=3)
    species = json.load(open(out)) if os.path.exists(out) else []
    existing = [(s["name"], s["desc"]) for s in species]
    print(f"have {len(species)} / want {n} -> {out}  (few-shot: {', '.join(nm for nm, _ in fewshot)})")
    for name, desc in generate(server, fewshot, n - len(species), existing):
        species.append({"name": name, "desc": desc})
        json.dump(species, open(out, "w"), indent=2)        # incremental save
        print(f"  [{len(species):4d}] {name} — {desc[:66]}")
    return species


def cmd_fewshot(n, out):
    species = run(SEED_ENTRIES, n, out)
    print(f"\ndone: {len(species)} candidates in {out}\n"
          f"-> pick your top ~3 favourites into species_fewshot.json, then: gen_species.py generate")


def cmd_generate(n, fewshot_path, out):
    if not os.path.exists(fewshot_path):
        sys.exit(f"no few-shot file at {fewshot_path} — run `gen_species.py fewshot` and curate it first.")
    fewshot = [(s["name"], s["desc"]) for s in json.load(open(fewshot_path))]
    species = run(fewshot, n, out)
    print(f"\ndone: {len(species)} species in {out} — now curate by hand (delete the duds).")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fewshot"
    rest = sys.argv[2:]
    if cmd == "fewshot":
        cmd_fewshot(int(rest[0]) if rest else 150, rest[1] if len(rest) > 1 else "fewshot_candidates.json")
    elif cmd == "generate":
        cmd_generate(int(rest[0]) if rest else 1000,
                     rest[1] if len(rest) > 1 else "species_fewshot.json",
                     rest[2] if len(rest) > 2 else "species_raw.json")
    else:
        sys.exit("usage: gen_species.py fewshot [N] [out] | generate [N] [fewshot.json] [out]")


if __name__ == "__main__":
    main()
