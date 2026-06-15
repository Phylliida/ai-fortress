"""
bake_item_needs.py — species-AGNOSTIC item->needs bake. For each of the 100 favorite items, which of the
recurring needs (>=2 species, from species_needs_clean.json) it fills, with refill amount. Two-step per
(item,need): yes/no gate then gen_percent degree. Writes item_needs.json incrementally.
"""
import json
from collections import Counter
import baseModelPrimitives as bmp
import needs

s = bmp.LlamaServer(timeout=60, retries=3)
items = json.load(open("items_favorites.json"))
recs = json.load(open("species_needs_clean.json"))

vocab = Counter()
for r in recs:
    for n in set(r["have"]) | set(r["extra"]):
        vocab[n] += 1
need_list = sorted((n for n, c in vocab.items() if c >= 2), key=lambda n: -vocab[n])

out = {}
for i, it in enumerate(items):
    name = it["name"]
    served = needs.bake_item_affordances(s, name, need_list)
    out[name] = served
    json.dump(out, open("item_needs.json", "w"), indent=2, ensure_ascii=False)
    top = ", ".join(f"{k}:{v}" for k, v in sorted(served.items(), key=lambda kv: -kv[1]))
    print(f"{i+1:3d}/{len(items)}  {name:22s} -> {top or '(nothing)'}", flush=True)
print(f"DONE -> item_needs.json ({len(need_list)} needs swept)")
