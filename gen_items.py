#!/usr/bin/env python3
"""
gen_items.py — generate an item/object list (name + description together) with GLM, in two steps.

Sibling of gen_species.py — same Monster-Manual one-pass machinery (each entry "Name: description" in a
single coherent generation), but framed as a CATALOGUE OF OBJECTS so it spans the whole sweep: raw
materials, plants/seeds/flowers, foods, tools, furnishings, machines, and stranger wares (right up to
cybernetics). Naming the object-subtypes in the header fences it to actual items (no animals/landscapes).

  STEP 1 — candidates from a tiny HAND-WRITTEN seed (bootstrap only):
      python3 gen_items.py fewshot [N=150] [out=item_candidates.json]
    Pick your top ~4-5 favourites (ideally spanning categories) into items_fewshot.json.

  STEP 2 — the polished list, few-shotting from your hand-picked entries:
      python3 gen_items.py generate [N=1000] [fewshot=items_fewshot.json] [out=items_raw.json]

Resamples one entry at a time (warm temp), dedups on the NAME with embeddinggemma (REQUIRED — errors if
it's down, never falls back), writes incrementally, resumable.
"""
import json
import os
import re
import sys
import baseModelPrimitives as bmp

# Hand-written bootstrap entries — used ONLY to seed STEP 1, then replaced by your base-model picks. The
# spread (ore -> seed -> flower -> food -> furniture -> electronics -> cybernetic) is what licenses the
# model to roam the full range of objects.
SEED_ENTRIES = [
    ("Iron ore", "Raw reddish rock veined with metal, smelted into tools and blades."),
    ("Wheat seeds", "A handful of pale grains, sown in spring to grow fields of bread-wheat."),
    ("Rose", "A thorned flower of velvet-red petals, prized for beauty and perfume."),
    ("Wheel of cheese", "A waxed round of aged cheese, sharp and crumbling, that keeps for months."),
    ("Oak writing desk", "A heavy hardwood desk with brass-handled drawers and an inkwell groove."),
    ("Television", "A glass-fronted box that flickers with moving pictures and tinny sound."),
    ("Neural implant", "A hair-thin chip laced into the spine, feeding data straight to the mind."),
]
HEADER = ("A catalogue of the world's objects — raw materials, plants, foods, tools, furnishings, "
          "machines, and stranger wares, one described per line:")


def manual_prompt(entries):
    """A catalogue the model continues; each line is 'Name: description', so one completion yields a NAME
    and a matching DESCRIPTION together. Naming the subtypes in HEADER keeps it to objects (not creatures
    or landscapes); 'machines'/'stranger wares' + the seed's TV/implant reach into tech and sci-fi."""
    return HEADER + "\n\n" + "".join(f"{n[:1].upper() + n[1:]}: {d}\n" for n, d in entries)


def parse_entry(raw):
    if ":" not in raw:
        return None
    name, desc = raw.split(":", 1)
    name, desc = name.strip(), desc.strip()
    desc = desc[:1].upper() + desc[1:]
    return (name, desc) if name and desc else None


def _reject(name):
    """Drop garbled names. Looser word-limit than species — item names run longer ('wheel of cheese')."""
    return (bool(re.search(r'[:;,\d"*|()_]', name))
            or len(name.split()) > 5
            or any(ord(c) > 0x024F for c in name))


def _key(name):
    return re.sub(r"\s+", "", name.lower())


def generate(server, fewshot, n, existing, sim_threshold=0.80):
    """Yield (name, desc): resample the catalogue prompt, parse one entry, dedup on the NAME. Slightly
    higher sim_threshold than species — item names share more words ('iron ore'/'copper ore')."""
    embs, seen = [], set()
    for name, _ in existing:
        seen.add(_key(name))
        embs.append(bmp.embed_texts([name])[0])
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
        e = bmp.embed_texts([name])[0]                     # no fallback — if embeddinggemma dies, crash loud
        if embs and max(bmp.cosine(e, pe) for pe in embs) >= sim_threshold:
            continue
        seen.add(_key(name))
        embs.append(e)
        got += 1
        yield name, desc


def run(fewshot, n, out):
    server = bmp.LlamaServer(timeout=60, retries=3)
    try:
        bmp.embed_texts(["ping"])      # dedup REQUIRES embeddinggemma — fail loud upfront, never fall back
    except Exception as e:
        sys.exit(f"embeddinggemma ({bmp.EMBED_URL}) is down — start it first "
                 f"(llama-server -m embeddinggemma-300M-Q8_0.gguf --embedding --pooling mean --port 8062). "
                 f"[{type(e).__name__}]")
    items = json.load(open(out)) if os.path.exists(out) else []
    existing = [(s["name"], s["desc"]) for s in items]
    print(f"have {len(items)} / want {n} -> {out}  (few-shot: {', '.join(nm for nm, _ in fewshot)})")
    for name, desc in generate(server, fewshot, n - len(items), existing):
        items.append({"name": name, "desc": desc})
        json.dump(items, open(out, "w"), indent=2)
        print(f"  [{len(items):4d}] {name} — {desc[:64]}")
    return items


def cmd_fewshot(n, out):
    items = run(SEED_ENTRIES, n, out)
    print(f"\ndone: {len(items)} candidates in {out}\n"
          f"-> pick ~4-5 favourites (spanning categories) into items_fewshot.json, then: gen_items.py generate")


def cmd_generate(n, fewshot_path, out):
    if not os.path.exists(fewshot_path):
        sys.exit(f"no few-shot file at {fewshot_path} — run `gen_items.py fewshot` and curate it first.")
    fewshot = [(s["name"], s["desc"]) for s in json.load(open(fewshot_path))]
    items = run(fewshot, n, out)
    print(f"\ndone: {len(items)} items in {out} — now curate by hand (delete the duds).")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fewshot"
    rest = sys.argv[2:]
    if cmd == "fewshot":
        cmd_fewshot(int(rest[0]) if rest else 150, rest[1] if len(rest) > 1 else "item_candidates.json")
    elif cmd == "generate":
        cmd_generate(int(rest[0]) if rest else 1000,
                     rest[1] if len(rest) > 1 else "items_fewshot.json",
                     rest[2] if len(rest) > 2 else "items_raw.json")
    else:
        sys.exit("usage: gen_items.py fewshot [N] [out] | generate [N] [fewshot.json] [out]")


if __name__ == "__main__":
    main()
