"""
rulesTests.py — DEBUG-ONLY calibration/validation harness for the base-model primitives and the
needs rules. NOT imported by prod. Run:

    python3 rulesTests.py              # pass/fail summary; rationale shown only for failures
    python3 rulesTests.py --reasoning  # rationale shown for every case

Validates yes_no_prob (need-applies) and gen_percent (satisfaction) against labeled expectations
across a broad range — humans, animals, plants, machines, inanimate objects, fictional beings.
The point is to catch calibration regressions as we tune prompts/scales, and (via --reasoning)
to see WHY the model judged the way it did when a case looks wrong.
"""
import sys
import baseModelPrimitives as bmp
import needs

REASON_ALL = "--reasoning" in sys.argv
S = bmp.LlamaServer(timeout=120, retries=3)

# (species, need, expected_applies)
NEED_CASES = [
    ("human", "food", True), ("human", "sleep", True), ("human", "water", True),
    ("human", "social", True), ("human", "bathroom", True), ("human", "love", True),
    ("dog", "food", True), ("dog", "sleep", True), ("dog", "social", True),
    ("fish", "water", True), ("fish", "air", False),
    ("horse", "food", True), ("horse", "movement", True),
    ("plant", "water", True), ("plant", "sleep", False), ("plant", "social", False),
    ("robot", "food", False), ("robot", "sleep", False), ("robot", "water", False),
    ("robot", "bathroom", False),
    ("rock", "food", False), ("rock", "water", False), ("rock", "sleep", False),
    ("wooden table", "food", False), ("wooden chair", "sleep", False),
    ("vampire", "food", True), ("vampire", "social", True), ("vampire", "bathroom", False),
    ("ghost", "food", False), ("ghost", "sleep", False),
    ("dragon", "food", True), ("dragon", "sleep", True),
    # a zombie's drive is for *flesh* (a species-specific extra), not a regular food core-need:
    ("zombie", "food", False),
]


def sat_prompt(activity, need):
    return f"Q: How much does {activity} satisfy someone's {need}?\nA: It satisfies their {need}"


# (activity, felt-need, lo, hi)  — expected gen_percent value within [lo, hi]
SAT_CASES = [
    ("eating a large hearty meal", "hunger", 0.55, 1.0),
    ("eating a huge feast", "hunger", 0.65, 1.0),
    ("eating a single small grape", "hunger", 0.0, 0.4),
    ("drinking a tall glass of water", "thirst", 0.55, 1.0),
    ("taking one tiny sip of water", "thirst", 0.05, 0.55),
    ("sleeping for a full eight hours", "tiredness", 0.65, 1.0),
    ("a quick five-minute nap", "tiredness", 0.1, 0.65),
    ("a long evening with close friends", "loneliness", 0.5, 1.0),
    ("a warm relaxing bath", "dirtiness", 0.5, 1.0),
    ("glancing at a blank wall", "hunger", 0.0, 0.35),
    ("staring at a brick", "thirst", 0.0, 0.35),
]


def run_need_cases():
    print("NEED-APPLIES  (yes_no_prob, threshold 0.5)")
    passed = 0
    for sp, nd, exp in NEED_CASES:
        prompt = needs.need_applies_prompt(sp, nd)
        why = None
        if REASON_ALL:
            r = S.yes_no_prob(prompt, reasoning=True)
            p, why = r["p"], r["why"]
        else:
            p = S.yes_no_prob(prompt)
        got = p >= 0.5
        ok = got == exp
        passed += ok
        print(f"  {'OK ' if ok else 'XX '} {sp + '/' + nd:26s} "
              f"exp={'yes' if exp else 'no':3s} got={'yes' if got else 'no':3s} p={p:.2f}")
        if not ok and why is None:
            why = S.yes_no_prob(prompt, reasoning=True)["why"]
        if why:
            print(f"        ↳ {why}")
    print(f"  {passed}/{len(NEED_CASES)} passed\n")
    return passed, len(NEED_CASES)


def run_sat_cases():
    print("SATISFACTION  (gen_percent)")
    passed = 0
    for act, nd, lo, hi in SAT_CASES:
        prompt = sat_prompt(act, nd)
        r = S.gen_percent(prompt, reasoning=REASON_ALL)
        v = r["value"]
        ok = lo <= v <= hi
        passed += ok
        print(f"  {'OK ' if ok else 'XX '} {act[:34]:34s} -> {nd:10s} {v:.2f} in [{lo:.2f},{hi:.2f}]")
        why = r.get("why")
        if not ok and why is None:
            why = S.gen_percent(prompt, reasoning=True)["why"]
        if why:
            print(f"        ↳ {why}")
    print(f"  {passed}/{len(SAT_CASES)} passed\n")
    return passed, len(SAT_CASES)


if __name__ == "__main__":
    n_pass, n_tot = run_need_cases()
    s_pass, s_tot = run_sat_cases()
    print(f"TOTAL: {n_pass + s_pass}/{n_tot + s_tot} passed")
