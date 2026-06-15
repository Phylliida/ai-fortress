"""
item_needs_view.py — read item_needs.json and show what the species-agnostic item->needs bake produced:
the INVERSE map (need -> items that fill it, the "what fills hunger?" view), items that fill nothing, and
which recurring needs turned out item-fillable vs creature-only.
"""
import json
from collections import Counter, defaultdict

item_needs = json.load(open("item_needs.json"))
recs = json.load(open("species_needs_clean.json"))
vocab = Counter()
for r in recs:
    for n in set(r["have"]) | set(r["extra"]):
        vocab[n] += 1
recurring = [n for n, c in vocab.items() if c >= 2]

# inverse: need -> [(item, refill)]
fills = defaultdict(list)
for item, served in item_needs.items():
    for need, refill in served.items():
        fills[need].append((item, refill))

print(f"=== {len(item_needs)} items baked ===\n")
print("NEED <- items that fill it (refill), recurring needs by species-frequency:")
for need in sorted(recurring, key=lambda n: -vocab[n]):
    lst = sorted(fills.get(need, []), key=lambda kv: -kv[1])
    tag = "" if lst else "   <-- creature-only (no item fills it)"
    shown = ", ".join(f"{it}({rf})" for it, rf in lst[:8]) + (" ..." if len(lst) > 8 else "")
    print(f"  {need:14s} (sp{vocab[need]:3d}, {len(lst):2d} items){tag}  {shown}")

nothing = [it for it, s in item_needs.items() if not s]
print(f"\nitems that fill NOTHING ({len(nothing)}): {', '.join(nothing)}")

item_fillable = [n for n in recurring if fills.get(n)]
creature_only = [n for n in recurring if not fills.get(n)]
print(f"\nitem-fillable needs: {len(item_fillable)}/{len(recurring)}")
print(f"creature-only needs ({len(creature_only)}): {', '.join(creature_only)}")
avg = sum(len(s) for s in item_needs.values()) / max(len(item_needs), 1)
print(f"avg needs filled per item: {avg:.1f}")
