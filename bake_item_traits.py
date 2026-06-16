"""
bake_item_traits.py — bake the traits.py physical traits (weight/size/rarity/worth/source/emission) over all
116 items (favorites + utilities), to fold in as extra columns alongside the need-affordances for re-factoring.
Writes item_traits.json incrementally.
"""
import json
import baseModelPrimitives as bmp
import traits

s = bmp.LlamaServer(timeout=60, retries=3)
items = json.load(open("items_favorites.json")) + json.load(open("items_utility.json"))

out = {}
for i, it in enumerate(items):
    name = it["name"]
    t = traits.bake_traits(s, name, samples=5)
    out[name] = t
    json.dump(out, open("item_traits.json", "w"), indent=2, ensure_ascii=False)
    em = "+".join(t["emission"]) or "-"
    print(f"{i+1:3d}/{len(items)}  {name:20s} wt={t['weight_g']:>9g}g size={t['size_cm']:>7g}cm "
          f"rarity={t['rarity']:>4} worth={t['worth']:>4} src={t['source']:11s} emit={em}", flush=True)
print(f"DONE -> item_traits.json ({len(out)} items)")
