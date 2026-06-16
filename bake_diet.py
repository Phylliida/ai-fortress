"""
bake_diet.py — diet-type over all 150 species, EVERY species queried via the base model (gen_categorical
distribution). No kingdom shortcut: a carnivorous/parasitic plant must be free to come back carnivore, not
forced to photosynthetic (Venus flytrap -> both, dodder -> plants). Real species use the bare name (the
distribution read argmaxes correctly even when ambiguous); fantasy use name + description. kingdom is kept
as metadata only. Writes species_diet.json incrementally.
"""
import json
from collections import Counter, defaultdict
import baseModelPrimitives as bmp
import needs

s = bmp.LlamaServer(timeout=60, retries=3)
real = json.load(open("species_real.json"))
fav = json.load(open("species_favorites.json"))

out = {}


def save():
    json.dump(out, open("species_diet.json", "w"), indent=2, ensure_ascii=False)


for r in real:
    d = needs.classify_diet(s, r["name"]); d["kingdom"] = r["kingdom"]; out[r["name"]] = d
    save()
    print(f"  {r['name']:22s} [{r['kingdom']:7s}] {d['diet']:14s} {d['conf']}{'  [unsure]' if d['unsure'] else ''}", flush=True)
for r in fav:
    d = needs.classify_diet(s, r["name"], r.get("desc")); d["kingdom"] = "Fantasy"; out[r["name"]] = d
    save()
    print(f"  {r['name']:22s} [Fantasy] {d['diet']:14s} {d['conf']}{'  [unsure]' if d['unsure'] else ''}", flush=True)

print("\ndiet distribution:", dict(Counter(v["diet"] for v in out.values())))
ct = defaultdict(Counter)
for v in out.values():
    ct[v["kingdom"]][v["diet"]] += 1
print("kingdom x diet (kingdom is metadata only — diet came from the model):")
for k in ["Animals", "Plants", "Fungi", "Fantasy"]:
    if k in ct:
        print(f"  {k:8s} {dict(ct[k])}")
print("non-photosynthetic PLANTS (exceptions the kingdom-rule would have hidden):",
      [k for k, v in out.items() if v["kingdom"] == "Plants" and v["diet"] != "photosynthetic"])
print("unsure:", [k for k, v in out.items() if v.get("unsure")])
print(f"DONE -> species_diet.json ({len(out)} species)")
