"""
bake_diet.py — diet-type over all 150 species. Plants/Fungi assigned from kingdom (definitional:
photosynthetic/decomposer, no query); Animals + Fantasy classified via gen_categorical (distribution
argmax + confidence). Writes species_diet.json incrementally. Diet gates EATING affordances downstream.
"""
import json
from collections import Counter
import baseModelPrimitives as bmp
import needs

s = bmp.LlamaServer(timeout=60, retries=3)
real = json.load(open("species_real.json"))
fav = json.load(open("species_favorites.json"))


def disambig(r):
    k = r.get("kingdom")
    return r["name"] + (" plant" if k == "Plants" else " fungus" if k == "Fungi" else "")


out = {}


def save():
    json.dump(out, open("species_diet.json", "w"), indent=2, ensure_ascii=False)


for r in real:
    k = r["kingdom"]
    if k == "Plants":
        out[r["name"]] = {"diet": "photosynthetic", "conf": 1.0, "unsure": False, "src": "kingdom"}
    elif k == "Fungi":
        out[r["name"]] = {"diet": "decomposer", "conf": 1.0, "unsure": False, "src": "kingdom"}
    else:
        d = needs.classify_diet(s, disambig(r)); d["src"] = "query"; out[r["name"]] = d
    save()
    v = out[r["name"]]
    print(f"  {r['name']:22s} {v['diet']:14s} {v['conf']}", flush=True)
for r in fav:
    d = needs.classify_diet(s, r["name"], r.get("desc")); d["src"] = "query"; out[r["name"]] = d
    save()
    print(f"  {r['name']:22s} {d['diet']:14s} {d['conf']}{'  [unsure]' if d['unsure'] else ''}", flush=True)

print("\ndiet distribution:", dict(Counter(v["diet"] for v in out.values())))
print("unsure (cross-cutting/exotic):", [k for k, v in out.items() if v.get("unsure")])
print(f"DONE -> species_diet.json ({len(out)} species)")
