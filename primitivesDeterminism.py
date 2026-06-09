"""
primitivesDeterminism.py — DEBUG-ONLY guardrail tests for the base-model primitives. NOT
imported by prod. Run this after switching base model or llama.cpp version (or touching
LlamaServer) to catch regressions like the grammar-forced-logprob non-determinism bug.

    python3 primitivesDeterminism.py      # exits nonzero if any check fails

Three groups:
  A. SCHEMA       — the llama-server response shape the primitives DEPEND on (top_logprobs must
                    carry 'id'+'logprob'; /tokenize?with_pieces pieces must join back to the
                    text; /detokenize must round-trip). A silent schema change here is what made
                    the by-id lookup miss before.
  B. DETERMINISM  — the LOGPROB primitives (yes_no_prob, gen_percent, _phrase_logprob) must return
                    IDENTICAL values across repeated calls. This is exactly where grammar-forcing
                    broke: it mis-reported the chosen token's logprob bimodally. (gen_number is
                    deliberately NOT here — it SAMPLES, so it is non-deterministic by design.)
  C. SANITY       — values in [0,1]; a known anchor in the right ballpark, so a model swap that
                    silently destroys calibration is also caught.
"""
import sys
import requests
import baseModelPrimitives as bmp

S = bmp.LlamaServer(timeout=120, retries=3)
REPS = 4
TOL = 1e-3                      # the bug spread ~0.4 (value) / ~15 (logprob); float noise << this
fails = []


def check(name, ok, detail=""):
    print(f"  {'OK ' if ok else 'XX '} {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def spread(vals):
    return max(vals) - min(vals)


P = lambda a, n: f"Q: How much does {a} satisfy someone's {n}?\nA: It satisfies their {n}"
BRICK, BREAD = P("staring at a brick", "thirst"), P("eating a loaf of bread", "hunger")
YN = "If a human were real, would it have a food need? Answer yes or no.\nAnswer:"

# ---------------------------------------------------------------- A. schema
print("A. SCHEMA (server response shape the primitives depend on)")
pieces = S.tokenize_pieces(" a lot")
check("tokenize?with_pieces -> (int id, str piece)", all(isinstance(i, int) and isinstance(p, str) for i, p in pieces))
check("pieces join back to the text", "".join(p for _, p in pieces) == " a lot",
      repr("".join(p for _, p in pieces)))
ids = [i for i, _ in pieces]
det = requests.post(S.url + "/detokenize", json={"tokens": ids}, timeout=30).json().get("content")
check("/detokenize round-trips", det == " a lot", repr(det))
out = S.complete("The capital of France is", n_predict=1, n_probs=40)
tl = (out.get("completion_probabilities") or [{}])[0].get("top_logprobs") or []
check("top_logprobs present + nonempty", len(tl) > 0, f"{len(tl)} candidates")
check("top_logprobs entries carry 'id'", bool(tl) and all("id" in c for c in tl))
check("top_logprobs entries carry 'logprob'", bool(tl) and all("logprob" in c for c in tl))

# ---------------------------------------------------------------- B. determinism
print("\nB. DETERMINISM (logprob primitives identical across calls; TOL=%g)" % TOL)
yn = [S.yes_no_prob(YN) for _ in range(REPS)]
check("yes_no_prob deterministic", spread(yn) < TOL, f"spread {spread(yn):.2e}")
gp = [S.gen_percent(BRICK)["value"] for _ in range(REPS)]          # brick = the case that exposed the bug
check("gen_percent deterministic", spread(gp) < TOL, f"spread {spread(gp):.2e}  vals~{gp[0]:.4f}")
pl = [S._phrase_logprob(BRICK, " a lot") for _ in range(REPS)]
check("_phrase_logprob deterministic", spread(pl) < TOL, f"spread {spread(pl):.2e}  val~{pl[0]:.3f}")

# ---------------------------------------------------------------- C. sanity
print("\nC. RANGE / SANITY")
bread = S.gen_percent(BREAD)["value"]
check("yes_no_prob in [0,1]", 0.0 <= yn[0] <= 1.0)
check("gen_percent in [0,1]", 0.0 <= bread <= 1.0 and 0.0 <= gp[0] <= 1.0)
check("filling food scores high (bread>hunger > 0.5)", bread > 0.5, f"{bread:.2f}")
check("irrelevant pair scores low (brick>thirst < 0.4)", gp[0] < 0.4, f"{gp[0]:.2f}")
num = S.gen_number("Q: How many legs does a typical dog have?\nA:")          # gen_number parses (sampled)
check("gen_number parses a clear case", num is not None and 2 <= num.minVal <= 8,
      None if num is None else f"{num.minVal}-{num.maxVal}")

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
