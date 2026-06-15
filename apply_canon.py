"""
apply_canon.py — apply need_canon.json (merge-map + drop-list) to the discovered needs.

Rewrites each species' have/extra through the canonical map, drops junk tokens, and dedups within the
species (a merge can collapse two of its needs into one). Writes species_needs_clean.json and prints the
before/after vocabulary shrink + the cleaned distribution.
"""
import json
from collections import defaultdict

canon = json.load(open("need_canon.json"))
rev = {}  # variant -> canonical
for canonical, variants in canon["merge"].items():
    for v in variants:
        rev[v] = canonical
drop = set(canon["drop"])


def clean(need_list):
    out = []
    for n in need_list:
        n = rev.get(n, n)        # merge to canonical if listed
        if n in drop or n in out:  # drop junk; dedup within species
            continue
        out.append(n)
    return out


recs = []
for tag in ["real", "fantasy"]:
    recs += json.load(open(f"species_needs_{tag}.json"))

before = set()
for r in recs:
    before |= set(r["have"]) | set(r["extra"])

for r in recs:
    r["have"] = clean(r["have"])
    r["extra"] = clean([e for e in r["extra"] if e not in r["have"]])  # extra shouldn't repeat a (now-merged) core

json.dump(recs, open("species_needs_clean.json", "w"), indent=2, ensure_ascii=False)

vocab = defaultdict(int)
for r in recs:
    for n in set(r["have"]) | set(r["extra"]):
        vocab[n] += 1

print(f"vocabulary: {len(before)} -> {len(vocab)} distinct needs "
      f"({len(canon['merge'])} canonical merges, {len(drop)} dropped)\n")
print("cleaned distribution (need : species count), needs in >=2 species:")
multi = sorted(((n, c) for n, c in vocab.items() if c >= 2), key=lambda kv: -kv[1])
for n, c in multi:
    print(f"  {n:16s} {c}")
singles = sorted(n for n, c in vocab.items() if c == 1)
print(f"\nkept singletons ({len(singles)}):\n  {', '.join(singles)}")
