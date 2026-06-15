"""
curate_needs.py — build the curation view from species_needs_{real,fantasy}.json.

Surfaces what to hand-check before the affordance bake: the global need vocabulary (each need -> how many
species, core vs extra, examples), and flags — empty-core species, singleton needs (likely junk or
over-specific), and near-duplicate needs the embed-dedup missed (sun/sunlight, company/companionship).
"""
import json
from collections import defaultdict

sets, allrecs = [], []
for tag in ["real", "fantasy"]:
    try:
        rs = json.load(open(f"species_needs_{tag}.json")); sets.append((tag, len(rs))); allrecs += rs
    except FileNotFoundError:
        pass

vocab = defaultdict(list)  # need -> [(species, kingdom, 'core'|'extra')]
for r in allrecs:
    for n in r["have"]:  vocab[n].append((r["name"], r["kingdom"], "core"))
    for n in r["extra"]: vocab[n].append((r["name"], r["kingdom"], "extra"))

print(f"=== {' + '.join(f'{n} {t}' for t,n in sets)} = {len(allrecs)} species, {len(vocab)} distinct needs ===\n")
print("NEED VOCABULARY  (need : total | core/extra | examples)")
for need, lst in sorted(vocab.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    nc = sum(1 for _, _, t in lst if t == "core")
    egs = ", ".join(s for s, _, _ in lst[:5])
    print(f"  {need:20s} {len(lst):3d}  (core {nc:3d} / extra {len(lst)-nc:3d})   {egs}")

print("\n=== FLAGS ===")
empty = [r["name"] for r in allrecs if not r["have"]]
print(f"\nempty core ({len(empty)}):\n  {', '.join(empty) or '(none)'}")

singletons = sorted(n for n, l in vocab.items() if len(l) == 1)
print(f"\nsingleton needs ({len(singletons)}) — likely junk / over-specific:\n  {', '.join(singletons)}")

nl = sorted(vocab)
dups = [f"{a}~{b}" for i, a in enumerate(nl) for b in nl[i+1:]
        if a in b or b in a or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5])]
print(f"\npossible near-dups to merge ({len(dups)}):\n  {', '.join(dups) or '(none)'}")
