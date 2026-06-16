"""
factor_combined.py — re-factor items with PHYSICAL TRAITS folded in as columns next to need-affordances.
Asks: (a) do traits correlate with the need-categories (food <-> grown/hunted/light)? (b) do the need-LESS
items (Anvil, Gold bar) form their own trait-defined category (mined/crafted + heavy + valuable)?
"""
import json
from collections import Counter

NEED_THRESH = 0.5
item_needs = json.load(open("item_needs.json"))
item_traits = json.load(open("item_traits.json"))


def wbin(g):
    for hi, lab in [(10, "weightless"), (1000, "light"), (10000, "medium"), (100000, "heavy")]:
        if g < hi:
            return "wt:" + lab
    return "wt:massive"


def sbin(cm):
    for hi, lab in [(5, "tiny"), (50, "small"), (200, "medium"), (1000, "large")]:
        if cm < hi:
            return "sz:" + lab
    return "sz:huge"


def band(v, labels, a=3.5, b=6.5):
    return labels[0] if v < a else (labels[1] if v < b else labels[2])


def trait_cols(t):
    c = {wbin(t["weight_g"]), sbin(t["size_cm"]),
         "rare:" + band(t["rarity"], ("common", "uncommon", "rare")),
         "worth:" + band(t["worth"], ("cheap", "valuable", "precious")),
         "src:" + t["source"]}
    c |= {"emit:" + ch for ch in t.get("emission", {})}
    return c


items, need_cols_of = {}, {}
for name, served in item_needs.items():
    nc = {f"need:{n}" for n, v in served.items() if v >= NEED_THRESH}
    tc = trait_cols(item_traits[name]) if name in item_traits else set()
    items[name] = frozenset(nc | tc)
    need_cols_of[name] = nc
G = list(items)
attrs = set().union(*items.values())
print(f"{len(G)} items, {len(attrs)} attributes ({sum(a.startswith('need:') for a in attrs)} needs + {sum(not a.startswith('need:') for a in attrs)} trait-bins)\n")


def extent(B):
    B = frozenset(B)
    return frozenset(it for it in G if B <= items[it])


# FCA closed concepts
intents = set(items.values())
queue = list(intents)
while queue:
    a = queue.pop()
    for b in list(intents):
        c = a & b
        if c and c not in intents:
            intents.add(c)
            queue.append(c)
concepts = sorted(((B, extent(B)) for B in intents if len(extent(B)) >= 3),
                  key=lambda ce: -len(ce[1]) * len(ce[0]))

mixed = [(B, E) for B, E in concepts
         if any(x.startswith("need:") for x in B) and any(not x.startswith("need:") for x in B)]
print("(a) NEED+TRAIT concepts — do physical traits ride along with the need-categories?")
for B, E in mixed[:14]:
    print(f"  [{len(E):3d}] {{{', '.join(sorted(B))}}}  e.g. {', '.join(sorted(E)[:5])}")

puretrait = [(B, E) for B, E in concepts if all(not x.startswith("need:") for x in B) and len(B) >= 2]
print("\n(b) PURE-TRAIT concepts — categories that exist on traits alone (incl. the need-less items):")
for B, E in puretrait[:14]:
    print(f"  [{len(E):3d}] {{{', '.join(sorted(B))}}}  e.g. {', '.join(sorted(E)[:5])}")

# the previously need-less items: what do their traits look like?
needless = [n for n in G if not need_cols_of[n]]
print(f"\n(c) the {len(needless)} need-less items, by trait profile:")
for n in needless:
    print(f"  {n:18s} {', '.join(sorted(x for x in items[n]))}")
