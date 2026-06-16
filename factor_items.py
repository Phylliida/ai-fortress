"""
factor_items.py — factor the item->needs affordance matrix (item_needs.json) to see if higher-level
categories fall out (FACTORING.md). Pure Python (numpy is broken in this venv). Two reads:

  1. FCA — closed concepts: each is a maximal (item-set, shared-need-set). The shared-need-set IS a
     candidate trait/category; the lattice orders them general->specific.
  2. MDL-ish compression — greedily pick K trait-templates, describe each item as nearest-template +
     diffs, and compare bits to the raw matrix. High compression => a few categories predict affordances
     => we can query the CATEGORY instead of every (item,need) pair.
"""
import json
from collections import defaultdict

THRESH = 0.5  # "strongly fills" — drop borderline gate-passes (e.g. novelty's weak 0.4s)
item_needs = json.load(open("item_needs.json"))
items = {it: frozenset(n for n, v in served.items() if v >= THRESH) for it, served in item_needs.items()}
items = {it: ns for it, ns in items.items() if ns}  # drop items with no strong affordance
G = list(items)
allneeds = sorted(set().union(*items.values()))
print(f"{len(items)}/{len(item_needs)} items have a strong affordance (refill>={THRESH}); {len(allneeds)} needs in play\n")


def extent(B):
    B = frozenset(B)
    return frozenset(it for it in G if B <= items[it])


# ---- 1. FCA: closed intents via intersection-closure of the object intents ----
intents = set(items.values())
queue = list(intents)
while queue:
    a = queue.pop()
    for b in list(intents):
        c = a & b
        if c and c not in intents:
            intents.add(c)
            queue.append(c)

MIN_SUP = 3
concepts = sorted(((B, extent(B)) for B in intents if extent(B) and len(extent(B)) >= MIN_SUP),
                  key=lambda ce: (-len(ce[1]) * len(ce[0])))  # rank by "area" = support * trait-size
print(f"FCA: {len(concepts)} closed concepts with support>={MIN_SUP} (candidate categories), top 22 by area:")
for B, E in concepts[:22]:
    print(f"  [{len(E):3d} items x {len(B)} needs]  {{{'+'.join(sorted(B))}}}")
    print(f"         e.g. {', '.join(sorted(E)[:7])}{' ...' if len(E) > 7 else ''}")


# ---- 2. MDL-ish compression: greedily pick K trait-templates, item = nearest template + diffs ----
def cover_cost(templates):
    """bits ~ sum|template| (the shared trait tables) + sum over items min symmetric-diff to a template."""
    tcost = sum(len(t) for t in templates)
    dcost = 0
    for it in G:
        ns = items[it]
        dcost += min(len(ns ^ t) for t in templates) if templates else len(ns)
    return tcost + dcost, tcost, dcost


raw = sum(len(items[it]) for it in G)  # raw matrix = total 1s
cand = [B for B, E in concepts]
templates, best = [], raw
print(f"\nMDL-ish compression (raw matrix = {raw} ones):")
print(f"  {'K':>2}  {'total':>6}  {'traits':>6}  {'diffs':>6}  {'vs raw':>7}")
for k in range(1, 13):
    # greedily add the candidate template that most lowers total cost
    bestc, bestt = None, None
    for t in cand:
        if t in templates:
            continue
        c, _, _ = cover_cost(templates + [t])
        if bestc is None or c < bestc:
            bestc, bestt = c, t
    if bestt is None:
        break
    templates.append(bestt)
    tot, tc, dc = cover_cost(templates)
    print(f"  {k:>2}  {tot:>6}  {tc:>6}  {dc:>6}  {tot/raw:>6.0%}   +{{{'+'.join(sorted(bestt))}}}")
