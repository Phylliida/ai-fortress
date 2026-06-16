"""
layer1_join.py — the eating-affordance filter: diet (species) x food-type (item) gates the FOOD affordance
that the naive item-needs join would wrongly grant. Classifies food-type for the food-items (cached to
item_foodtype.json), then shows telling pairs and counts how many naive food-interactions the filter drops.
"""
import json
from collections import Counter
import baseModelPrimitives as bmp
import needs

s = bmp.LlamaServer(timeout=60, retries=3)
item_needs = json.load(open("item_needs.json"))
sp_needs = {r["name"]: r for r in json.load(open("species_needs_clean.json"))}
sp_diet = json.load(open("species_diet.json"))

FOOD_ITEMS = {it for it, served in item_needs.items() if served.get("food", 0) >= 0.5}
try:
    ftype = json.load(open("item_foodtype.json"))
except FileNotFoundError:
    ftype = {}
for it in sorted(FOOD_ITEMS):
    if it not in ftype:
        ftype[it] = needs.classify_food_type(s, it)
        json.dump(ftype, open("item_foodtype.json", "w"), indent=2, ensure_ascii=False)
        print(f"  food-type  {it:20s} {ftype[it]['food_type']} ({ftype[it]['conf']})", flush=True)


def has_food_need(sp):
    r = sp_needs.get(sp, {})
    return "food" in (r.get("have", []) + r.get("extra", []))


def affords_food(sp, it):
    if not (has_food_need(sp) and it in FOOD_ITEMS):
        return None
    return needs.can_eat(sp_diet.get(sp, {}).get("diet", "other"), ftype.get(it, {}).get("food_type", "other"))


print("\ndiet-filtered eating affordances:")
demo = [("Lion", "Strawberry"), ("Lion", "Sashimi platter"), ("Cattle", "Sashimi platter"),
        ("Cattle", "Baguette"), ("Human", "Strawberry"), ("Tomato", "Strawberry"), ("Orca", "Caviar")]
for sp, it in demo:
    r = affords_food(sp, it)
    diet = sp_diet.get(sp, {}).get("diet", "?"); ft = ftype.get(it, {}).get("food_type", "?")
    verdict = "EATS" if r else ("DROPPED" if r is False else "n/a")
    print(f"  {sp:10s}({diet:13s}) x {it:18s}({ft:7s}) -> {verdict}")

total = kept = 0
for sp in sp_needs:
    for it in FOOD_ITEMS:
        r = affords_food(sp, it)
        if r is None:
            continue
        total += 1; kept += 1 if r else 0
print(f"\nfood interactions: {total} naive-join -> {kept} kept, {total - kept} DROPPED by diet "
      f"({(total - kept) / max(total, 1):.0%}) — exceptions removed with zero per-pair queries")
