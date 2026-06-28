"""
exceptions.py — per-species eating EXCEPTIONS, both directions, the diff layer over the structural Layer-1
filter. A species PROPOSES exception categories (generative, high recall, noisy) which are then VERIFIED
adversarially (the propose step confabulates — "horses can't eat apples" — and only verify kills that):

  NEGATIVE: foods toxic/harmful to this species (dog -> chocolate, grapes, onion)   -> drop the affordance
  POSITIVE: unusual things this species CAN eat that most can't (termite -> wood)    -> add the affordance

Each verified category resolves to its items via categories.contained_in_category (compositional 'contains'/
'made_of'), so one generative pass per species + cheap category->item resolution, never the species×item grid.
"""
import baseModelPrimitives as bmp

# propose (multi-sample union for recall). vary the subject; the lists are species-specific knowledge.
TOXIC_FEWSHOT = (
    "Question: What foods are poisonous or dangerous for a dog to eat?\nAnswer: chocolate, grapes, onions, xylitol\n"
    "Question: What foods are poisonous or dangerous for a horse to eat?\nAnswer: avocado, chocolate, onions\n"
    "Question: What foods are poisonous or dangerous for a {species} to eat?\nAnswer:")
UNUSUAL_FEWSHOT = (
    "Question: What unusual things can a termite eat that most animals cannot?\nAnswer: wood, paper, cardboard\n"
    "Question: What unusual things can a goat eat that most animals cannot?\nAnswer: cardboard, thorny weeds, tin cans\n"
    "Question: What unusual things can a {species} eat that most animals cannot?\nAnswer:")

# verify (adversarial — kills the propose-step confabulations). Mixed Yes/No, high-perplexity order
# (Y N N Y — neither grouped nor strictly alternating), varied species+foods.
VERIFY_NEG_FEWSHOT = (
    "Question: Is chocolate genuinely toxic or dangerous for a dog to eat?\nAnswer: Yes\n"
    "Question: Is chicken genuinely toxic or dangerous for a dog to eat?\nAnswer: No\n"
    "Question: Is grass genuinely toxic or dangerous for a rabbit to eat?\nAnswer: No\n"
    "Question: Is onion genuinely toxic or dangerous for a cat to eat?\nAnswer: Yes\n")
VERIFY_POS_FEWSHOT = (
    "Question: Can a termite genuinely eat wood, which most animals cannot?\nAnswer: Yes\n"
    "Question: Can a rabbit genuinely eat steel, which most animals cannot?\nAnswer: No\n"
    "Question: Can a goat genuinely eat cardboard, which most animals cannot?\nAnswer: Yes\n"
    "Question: Can a sheep genuinely eat glass, which most animals cannot?\nAnswer: No\n")


def _sample_union(server, prompt, samples):
    seen, out = set(), []
    for _ in range(samples):
        for t in server.gen_text(prompt, stop=["\n"], n_predict=40).split(","):
            t = t.strip().lower().rstrip(".")
            if t and t not in seen and len(t.split()) <= 3:
                seen.add(t); out.append(t)
    return out


def propose_exceptions(server, species, desc=None, samples=3):
    """Candidate ± exception categories (noisy — verify before use)."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    return {"negative": _sample_union(server, ctx + TOXIC_FEWSHOT.format(species=species), samples),
            "positive": _sample_union(server, ctx + UNUSUAL_FEWSHOT.format(species=species), samples)}


def verify_exception(server, species, category, sign, desc=None):
    """Adversarial verify of one candidate. Returns P(real). sign in {'negative','positive'}."""
    ctx = f"{species} is {desc}.\n" if desc else ""
    if sign == "negative":
        q = (ctx + VERIFY_NEG_FEWSHOT +
             f"Question: Is {category} genuinely toxic or dangerous for a {species} to eat?\nAnswer:")
    else:
        q = (ctx + VERIFY_POS_FEWSHOT +
             f"Question: Can a {species} genuinely eat {category}, which most animals cannot?\nAnswer:")
    return server.yes_no_prob(q)


def species_exceptions(server, species, desc=None, threshold=0.5, samples=3):
    """Full pipeline: propose ± categories, keep only the ones that pass adversarial verify.
    Returns {negative:[(cat,p)...], positive:[(cat,p)...]} — verified exception categories per species."""
    cand = propose_exceptions(server, species, desc, samples)
    out = {}
    for sign in ("negative", "positive"):
        kept = []
        for cat in cand[sign]:
            p = verify_exception(server, species, cat, sign, desc)
            if p >= threshold:
                kept.append((cat, round(p, 3)))
        out[sign] = kept
    return out
