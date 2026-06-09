"""
needs.py — colony-sim need discovery (hybrid candidate model).

A fixed UNIVERSAL_CORE of single-word biological/psychological needs is yes/no-checked against
each SPECIES every time (a robot may not 'sleep', a fungus may not need a 'bathroom'), then
species-specific extra needs are generated and embed-deduped against the applying core. This
step finds WHICH needs a species has; rates/decay (gen_number) and object affordances
(gen_percent gated by a yes/no "applies?") come next. Built on baseModelPrimitives.

Needs are baked per SPECIES (shared by all members); how fast each decays is per-person (later).
"""
from baseModelPrimitives import iter_unique, LOCATION_GRAMMAR

# Fixed universal core: single-word needs swept (yes/no) against every species, every time.
UNIVERSAL_CORE = [
    "food", "water", "sleep", "air", "warmth", "bathroom", "hygiene", "health",
    "safety", "shelter", "social", "love", "respect", "purpose", "novelty", "movement",
]


def need_applies_prompt(species, need):
    # "if it were real" sidesteps the model refusing needs to fictional beings (it fixated on
    # "dragons aren't real"); the counterfactual is harmless for species that already are real.
    return (f"If a {species} were real, would it have a {need} need "
            f"(something it must regularly satisfy)? Answer yes or no.\nAnswer:")


def extra_needs_prompt(species, have):
    return (f"A {species} has needs such as: {', '.join(have)}.\n"
            f"List other needs a {species} has, each a single word:\n"
            f"Another single-word need a {species} has is:\n-")


def discover_core(server, species, threshold=0.5):
    """yes/no-sweep the whole UNIVERSAL_CORE against `species`; each entry keeps P(yes)."""
    out = []
    for need in UNIVERSAL_CORE:
        p = server.yes_no_prob(need_applies_prompt(species, need))
        out.append({"need": need, "applies": p >= threshold, "p": round(p, 3)})
    return out


def discover_needs(server, species, n_extra=6, threshold=0.5):
    """Full hybrid discovery: core sweep + dedup-suggested single-word species-specific extras."""
    core = discover_core(server, species, threshold)
    have = [c["need"] for c in core if c["applies"]]
    # extras dedup against the FULL core (not just the applying ones) so a dropped-but-borderline
    # core word can't resurface as an "extra"; reject non-single-word / non-alphabetic junk.
    extra = [e.lower() for e in iter_unique(
        server, extra_needs_prompt(species, have), n=n_extra, grammar=LOCATION_GRAMMAR,
        max_len=20, seed=UNIVERSAL_CORE,
        reject=lambda x: len(x.split()) != 1 or not x.replace("-", "").replace("'", "").isalpha())]
    return {"species": species, "core": core, "have": have, "extra": extra}
