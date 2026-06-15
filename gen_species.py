#!/usr/bin/env python3
"""
gen_species.py — scaffolding to generate a fantasy-species list with GLM, in two human-in-the-loop steps.

Base-model few-shots beat hand-written ones (in-distribution examples generalize better) AND hand-picked
ones beat raw ones — so we generate MANY base-model candidates and you keep your favourites as the few-shot:

  STEP 1 — candidate few-shot examples (curate these down to your favourites):
      python3 gen_species.py fewshot [N=24] [out=fewshot_candidates.json]
    Generates N candidate (name, description) pairs in the model's OWN voice — bootstrapped from a tiny
    hand-written seed of *names only*; every description is written by the model (no hand-written style).
    Hand-curate -> keep your favourites -> save as species_fewshot.json.

  STEP 2 — the full list, few-shotting from your hand-picked examples:
      python3 gen_species.py generate [N=60] [fewshot=species_fewshot.json] [out=species_raw.json]
    Curate the result into your final species list.

Both resample one species at a time (warm temp), semantic-dedup with embeddinggemma (iter_unique), write
incrementally (a kill won't lose progress), and are resumable (re-running tops up to N).
"""
import json
import os
import re
import sys
import baseModelPrimitives as bmp

# Hand-written seeds — NAMES ONLY, used solely to bootstrap STEP 1's candidate generation, then dropped.
# There are NO hand-written descriptions anywhere; the model writes every description in its own voice.
SEED_SPECIES = ["goblin", "treant", "basilisk", "merfolk", "gargoyle", "wyvern"]


def species_prompt(examples):
    return ("A sprawling fantasy world is home to countless species — peoples, beasts, monsters, "
            "spirits, and constructs.\nExamples: " + ", ".join(examples) + ".\n\n"
            "Name one more fantasy species: ")


def describe(server, name, exemplars):
    """One-line bestiary description. exemplars = (name, desc) pairs for few-shot; pass [] to let the
    model write in its own voice (just a title prime — used while generating candidates)."""
    shots = "".join(f"{n[:1].upper() + n[1:]}: {d}\n" for n, d in exemplars)
    return server.gen_text("A fantasy bestiary, one line each.\n\n" + shots +
                           f"{name[:1].upper() + name[1:]}:", stop=["\n"], n_predict=56).strip()


def _reject(name):
    """Drop obvious degenerate samples (the curator shouldn't have to)."""
    return (bool(re.search(r'[:;,\d"*|()_]', name))        # punctuation/markdown that signals garbled output
            or len(name.split()) > 3                        # real species names are short
            or any(ord(c) > 0x024F for c in name))          # non-Latin script (GLM occasionally drifts)


def _run(server, prompt_examples, desc_exemplars, n, out, species, seed_extra=()):
    """Resample names off `prompt_examples`, describe each with `desc_exemplars`, save incrementally."""
    have = {s["name"].lower() for s in species}
    seed = [s["name"] for s in species] + list(seed_extra)
    for name in bmp.iter_unique(server, species_prompt(prompt_examples), n=max(0, n - len(species)),
                                seed=seed or list(SEED_SPECIES), max_len=30, reject=_reject):
        species.append({"name": name, "desc": describe(server, name, desc_exemplars)})
        have.add(name.lower())
        json.dump(species, open(out, "w"), indent=2)
        print(f"  [{len(species):3d}] {name} — {species[-1]['desc'][:64]}")


def cmd_fewshot(n, out):
    """STEP 1: a pool of base-model candidate (name, desc) pairs to hand-pick the few-shot from."""
    server = bmp.LlamaServer(timeout=60, retries=3)
    species = json.load(open(out)) if os.path.exists(out) else []
    have = {s["name"].lower() for s in species}
    print(f"generating {n} candidate few-shot examples (model writes its own descriptions) -> {out}")
    for name in SEED_SPECIES:                              # offer the well-known ones too (model-described)
        if name not in have and len(species) < n:
            species.append({"name": name, "desc": describe(server, name, [])})
            have.add(name)
            json.dump(species, open(out, "w"), indent=2)
            print(f"  [{len(species):3d}] {name} — {species[-1]['desc'][:64]}")
    _run(server, SEED_SPECIES, [], n, out, species)        # prompt seed = hand-written; descriptions = model-voice
    print(f"\ndone: {len(species)} candidates in {out}\n"
          f"-> hand-pick your favourites into species_fewshot.json, then run: gen_species.py generate")


def cmd_generate(n, fewshot_path, out):
    """STEP 2: the full list, few-shotting from your hand-picked base-model examples."""
    if not os.path.exists(fewshot_path):
        sys.exit(f"no few-shot file at {fewshot_path} — run `gen_species.py fewshot` and curate it first.")
    fs = json.load(open(fewshot_path))
    ex_names = [s["name"] for s in fs][:16]
    ex_descs = [(s["name"], s["desc"]) for s in fs][:6]
    server = bmp.LlamaServer(timeout=60, retries=3)
    species = json.load(open(out)) if os.path.exists(out) else []
    print(f"generating {n} species with {len(fs)} hand-picked few-shot examples -> {out}")
    _run(server, ex_names, ex_descs, n, out, species, seed_extra=ex_names)
    print(f"\ndone: {len(species)} species in {out} — now curate by hand (delete the duds).")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fewshot"
    rest = sys.argv[2:]
    if cmd == "fewshot":
        n = int(rest[0]) if rest else 24
        out = rest[1] if len(rest) > 1 else "fewshot_candidates.json"
        cmd_fewshot(n, out)
    elif cmd == "generate":
        n = int(rest[0]) if rest else 60
        fewshot = rest[1] if len(rest) > 1 else "species_fewshot.json"
        out = rest[2] if len(rest) > 2 else "species_raw.json"
        cmd_generate(n, fewshot, out)
    else:
        sys.exit("usage: gen_species.py fewshot [N] [out] | generate [N] [fewshot.json] [out]")


if __name__ == "__main__":
    main()
