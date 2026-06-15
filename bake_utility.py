"""
bake_utility.py — bake the mundane-utility items (items_utility.json) species-agnostically and MERGE into
item_needs.json, closing the basic-need coverage gaps the exotic favorites left (water/bathroom/hygiene/soil).
"""
import json
from collections import Counter
import baseModelPrimitives as bmp
import needs

s = bmp.LlamaServer(timeout=60, retries=3)
util = json.load(open("items_utility.json"))
recs = json.load(open("species_needs_clean.json"))
vocab = Counter()
for r in recs:
    for n in set(r["have"]) | set(r["extra"]):
        vocab[n] += 1
need_list = sorted((n for n, c in vocab.items() if c >= 2), key=lambda n: -vocab[n])

out = json.load(open("item_needs.json"))  # merge into the existing favorites bake
for i, it in enumerate(util):
    name = it["name"]
    served = needs.bake_item_affordances(s, name, need_list)
    out[name] = served
    json.dump(out, open("item_needs.json", "w"), indent=2, ensure_ascii=False)
    top = ", ".join(f"{k}:{v}" for k, v in sorted(served.items(), key=lambda kv: -kv[1]))
    print(f"{i+1:2d}/{len(util)}  {name:18s} -> {top or '(nothing)'}", flush=True)
print(f"DONE -> item_needs.json now {len(out)} items")
