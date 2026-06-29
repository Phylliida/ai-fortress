"""
categories.py — the "which items are in category C" primitive for the exception system, sub-linear.

Two regimes, auto-detected by probing membership along the embedding-sort:
  SURFACE (is-a: plant / bird / food) — membership is MONOTONE in similarity-to-category, so we embedding-sort
    and run ERROR-ROBUST binary search (probabilistic bisection, Horstein), using the Y/N logprob as the
    per-probe reliability. O(log N) Y/N queries.
  COMPOSITIONAL (contains-X: chocolate / onion / wood) — membership does NOT track whole-item similarity
    (mole sauce contains chocolate but embeds savory), so we extract each item's COMPONENTS once ("made
    from ...") and embed-match the category term against them. One extraction per item serves every
    compositional category.
The regime is detected by an inversion check: a low-similarity item that IS a member sitting below a
high-similarity item that ISN'T means whole-item similarity doesn't order membership -> compositional.
"""
import baseModelPrimitives as bmp

# --- component extraction ("made from"): ingredients for foods, materials for objects. Interleaved few-shot
#     (multi / single / multi — patterns not grouped, the single example not stranded last); now also with
#     Question:/Answer: labels + the counterfactual "if … were real" so made-up items (a magma sword) extract. ---
ING_FEWSHOT = (
    "Question: If a beef stew were real, what would it mainly be made from?\nAnswer: beef, potatoes, carrots, onion, broth\n"
    "Question: If a strawberry were real, what would it mainly be made from?\nAnswer: strawberry\n"
    "Question: If a chocolate chip cookie were real, what would it mainly be made from?\nAnswer: flour, butter, sugar, chocolate chips, eggs\n")


def extract_ingredients(server, item, samples=4):
    """Ingredients as a sample-union (server.sample_union): one draw sometimes drops the defining ingredient
    (mole sauce without 'chocolate'), the union reliably catches it. The embed-match takes the max over
    ingredients, so extra union terms don't hurt precision. NO verify on purpose — recall is strictly good for
    the compositional-category matching this feeds (see verified_ingredients for the display-side verified one)."""
    prompt = ING_FEWSHOT + f"Question: If a {item} were real, what would it mainly be made from?\nAnswer:"
    return server.sample_union(prompt, samples=samples, n_predict=50)


# display-side ingredient verify (the /item page wants a CLEAN list, unlike the recall-tuned category matching).
# Counterfactual for made-up items; mixed Y/N, high-perplexity (Y N N Y); cheese-free so verifying a wheel of
# cheese stays an honest test.
ING_VERIFY_FEWSHOT = (
    "Question: If a loaf of bread were real, would it be made from flour?\nAnswer: Yes\n"
    "Question: If a steel sword were real, would it be made from chocolate?\nAnswer: No\n"
    "Question: If a wool sweater were real, would it be made from plastic?\nAnswer: No\n"
    "Question: If a glass bottle were real, would it be made from sand?\nAnswer: Yes\n")


def ingredient_fits(server, item, ingredient, threshold=0.5):
    """Y/N: is `ingredient` really a main ingredient/material of `item`?"""
    q = f"If a {item} were real, would it be made from {ingredient}?"
    return server.yes_no_prob(ING_VERIFY_FEWSHOT + f"Question: {q}\nAnswer:") >= threshold


def verified_ingredients(server, item, samples=4, threshold=0.5):
    """extract_ingredients (recall) -> per-ingredient Y/N verify. For DISPLAY — drops hallucinated draws so the
    'made from' list is clean (the underlying extract_ingredients stays recall-only for category matching)."""
    return [x for x in extract_ingredients(server, item, samples=samples) if ingredient_fits(server, item, x, threshold)]


# --- surface membership Y/N. Generic "is X a {category}?", varied categories, high-perplexity answer
#     order (Y N Y Y N — neither grouped nor strictly alternating). ---
MEMBERSHIP_FEWSHOT = (
    "Question: Is a robin a bird?\nAnswer: Yes\n"
    "Question: Is an oak tree a bird?\nAnswer: No\n"
    "Question: Is a rose a plant?\nAnswer: Yes\n"
    "Question: Is a wrench a tool?\nAnswer: Yes\n"
    "Question: Is a river a tool?\nAnswer: No\n")


def membership_prompt(category, item):
    return MEMBERSHIP_FEWSHOT + f"Question: Is a {item} a {category}?\nAnswer:"


def _embed(texts, url):
    return bmp.embed_texts(texts, url)


def probabilistic_bisection(server, ranked, category, seed=None, max_queries=12):
    """Horstein's probabilistic bisection on `ranked` (sorted high->low membership). Belief over the boundary
    k in {0..N} (items[:k] are members); each probe at the belief-median item updates the belief using the
    Y/N logprob `p` as reliability (a 0.99 answer moves it hard, a 0.5 borderline barely). Returns cut k."""
    N = len(ranked)
    f = [1.0 / (N + 1)] * (N + 1)
    cache = dict(seed or {})

    def update(i, p):
        for k in range(N + 1):
            f[k] *= p if k > i else (1.0 - p)
        s = sum(f) or 1.0
        for k in range(N + 1):
            f[k] /= s

    for i, p in cache.items():          # fold in any seed probes
        update(i, p)
    for _ in range(max_queries):
        cum, med = 0.0, N
        for k in range(N + 1):
            cum += f[k]
            if cum >= 0.5:
                med = k
                break
        i = min(max(med, 0), N - 1)
        if i not in cache:
            cache[i] = server.yes_no_prob(membership_prompt(category, ranked[i]))
        update(i, cache[i])
        if max(f) > 0.9:                # belief concentrated -> done
            break
    return max(range(N + 1), key=lambda k: f[k])


def contained_in_category(server, items, category, relation="is_a", item_embs=None, components=None,
                          comp_threshold=0.72, embed_url=bmp.EMBED_URL):
    """Set of `items` in `category`, routed by RELATION (the exception generator emits it):
      relation='is_a'   -> SURFACE: embedding-sort by similarity-to-category + probabilistic bisection on
                           "is X a {category}?" (O(log N) Y/N).
      relation='contains'/'made_of' -> COMPOSITIONAL: extract each item's components once and embed-match
                           the category term against them (mole sauce -> chocolate even though it embeds savory).
    Pass precomputed `item_embs` / `components` (each {item: ...}) to amortize across many categories."""
    cat_emb = _embed([category], embed_url)[0]

    if relation in ("contains", "made_of", "ingredient"):
        if components is None:
            components = {it: extract_ingredients(server, it) for it in items}
        result = set()
        for it in items:
            comp = components.get(it) or []
            if comp and max(bmp.cosine(cat_emb, ce) for ce in _embed(comp, embed_url)) >= comp_threshold:
                result.add(it)
        return result

    # is_a -> surface: embedding-sort + probabilistic bisection, then VERIFY a band around the cut with the
    # Y/N (the embedding mis-orders near-boundary items — an insect that "flies" sits among birds). Items
    # well above the band are trusted in, well below trusted out. (A far mis-embedded member — a rare name
    # like "Kākāpō" that lands near the bottom — is the embedding's blind spot; band-verify can't reach it.)
    if item_embs is None:
        item_embs = {it: _embed([it], embed_url)[0] for it in items}
    ranked = sorted(items, key=lambda it: -bmp.cosine(item_embs[it], cat_emb))
    k = probabilistic_bisection(server, ranked, category)
    band = 3
    lo, hi = max(0, k - band), min(len(ranked), k + band)
    result = set(ranked[:lo])
    for it in ranked[lo:hi]:
        if server.yes_no_prob(membership_prompt(category, it)) >= 0.5:
            result.add(it)
    return result
