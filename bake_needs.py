"""
bake_needs.py — run need-discovery over a species family, saving per-species results incrementally.

Real species are disambiguated by KINGDOM ("Tomato" -> "Tomato plant", "Ergot" -> "Ergot fungus") because
bare crop/food names confuse the base model (it can't tell the plant from its edible form, so it projected
animal needs — breakfast/sleep — onto plants). Animals stay bare. Output: species_needs_<set>.json.
"""
import json, sys, baseModelPrimitives as bmp, needs

SET = sys.argv[1] if len(sys.argv) > 1 else "real"
src = {"real": "species_real.json", "fantasy": "species_favorites.json"}[SET]
recs = json.load(open(src))
out_path = f"species_needs_{SET}.json"


def disambig(rec):
    k = rec.get("kingdom")
    if k == "Plants": return f"{rec['name']} plant"
    if k == "Fungi": return f"{rec['name']} fungus"
    return rec["name"]


s = bmp.LlamaServer(timeout=60, retries=3)
results = []
for i, rec in enumerate(recs):
    sp = disambig(rec)
    desc = rec.get("desc")  # fantasy species carry a description to ground the invented name
    r = needs.discover_needs(s, sp, n_extra=6, desc=desc)
    r["name"] = rec["name"]; r["kingdom"] = rec.get("kingdom", "Fantasy"); r["queried_as"] = sp
    results.append(r)
    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)  # incremental save
    print(f"{i+1:3d}/{len(recs)}  {sp:28s} have=[{', '.join(r['have'])}]  extra=[{', '.join(r['extra'])}]", flush=True)
print(f"DONE -> {out_path}")
