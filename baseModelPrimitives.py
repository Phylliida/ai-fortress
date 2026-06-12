"""
baseModelPrimitives.py — the low-level base-model query primitives, separated from all
worldgen/domain code: the LlamaServer client + the three logprob primitives
(yes_no_prob / gen_number(_median) / gen_percent), plus gen_text / pick_from_set / gen_int /
gen_list, the GBNF grammar helpers, and the embeddinggemma helpers (embed_texts / cosine).

Talks HTTP to a hosted llama.cpp `llama-server` (SERVER_URL); no in-process model. Every
grammar ends with a newline terminator and callers stop on newline — that halts generation
the instant the grammar is satisfied, before the model can sample a stray special token that
underflows llama.cpp's grammar stack and crashes it.

Run a server first, e.g.:
  llama-server -m GLM-4-32B-Base.gguf -c 8192 --host 127.0.0.1 --port 8080
Embeddings (semantic dedup): llama-server -m embeddinggemma-300M-Q8_0.gguf --embedding ...
"""

import re
import json
import math
import time
import threading
import statistics
import requests

import numbers_parse as nums   # number parsing copied wholesale from worldcode.py


SERVER_URL = "http://172.22.146.1:8055"


# Explicit sampling sent on every request, overriding the server's configured sampler.
# temp 1 + top_k 0 + top_p 1 + min_p 0 + repeat_penalty 1 = the raw model distribution,
# no truncation or penalties.
DEFAULT_SAMPLING = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0}

# Local embeddinggemma server, for SEMANTIC dedup of generated locations:
#   llama-server -m embeddinggemma-300M-Q8_0.gguf --embedding --pooling mean --port 8062
EMBED_URL = "http://127.0.0.1:8062"


# --- base-model query log: every LlamaServer.complete() can be routed (per-thread) to a sink, so the
#     web UI can persist each prompt+output into the ACTIVE world's log for auditing/debugging.
_log_ctx = threading.local()


def set_log_sink(fn):
    """Route this thread's base-model queries to `fn(entry)` (None to stop). entry = {t, prompt,
    n_predict, grammar, content, top}. Set per-request by the web UI to the active world's log."""
    _log_ctx.sink = fn


def _emit_query_log(prompt, payload, result):
    sink = getattr(_log_ctx, "sink", None)
    if not sink:
        return
    try:
        top = []
        if (payload.get("n_predict") or 0) <= 2:                  # log the top tokens for score reads
            cp = result.get("completion_probabilities") or result.get("logprobs")
            for c in ((cp[0].get("top_logprobs") or [])[:6] if cp else []):
                lp = c.get("logprob")
                top.append([c.get("token") or c.get("tok") or "", round(math.exp(lp), 4) if lp is not None else None])
        sink({"t": time.time(), "prompt": prompt, "n_predict": payload.get("n_predict"),
              "grammar": bool(payload.get("grammar")), "content": (result.get("content") or "")[:400], "top": top})
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GBNF grammar helpers  (these replace the hand-rolled stop-lists / token tries)
# ---------------------------------------------------------------------------

def gbnf_escape(s: str) -> str:
    """Escape a literal for a GBNF double-quoted string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def alternation_grammar(options) -> str:
    """Force output to be exactly one of `options` (each with a LEADING space so it
    tokenizes naturally after 'Foo:', like YES_NO_GRAMMAR's " Yes"/" No"), then a "\\n"
    terminator. Terminator + stop=["\\n"] is what avoids the special-token crash."""
    alts = " | ".join(f'" {gbnf_escape(o.strip())}"' for o in options if o.strip())
    return f'root ::= ({alts}) "\\n"'


# Every grammar ENDS WITH A "\n" TERMINATOR, and callers pass stop=["\n"]. That halts
# generation the instant the grammar is satisfied — before the model can sample a stray
# special token (e.g. GLM-4's <|end_of_video|>) that underflows llama.cpp's grammar
# stack and crashes the server. (Confirmed fix; see LlamaServer.__init__.)
GRAMMAR_STOP = ["\n"]
YES_NO_GRAMMAR = 'root ::= (" Yes" | " No") "\\n"'
INT_GRAMMAR = 'root ::= "-"? [0-9]+ "\\n"'
INT_OR_RANGE_GRAMMAR = 'root ::= int (" - " int)? "\\n"\nint ::= "-"? [0-9]+'
# A location/object: any run of characters EXCEPT  /  "  ;  ,  .  -  (and newline), then terminator.
LOCATION_GRAMMAR = r'root ::= [^/";,.\n-]+ "\n"'

# A DURATION: "<number>[-<number>] <unit>" so the model answers in its NATURAL unit (a meal in
# minutes, a night in hours) instead of being forced into minutes (which makes it lowball things it
# thinks of in hours). gen_duration normalizes the unit to minutes. Terminator + stop=["\n"] as ever.
DURATION_GRAMMAR = (
    r'root ::= " "? num " "? unit "\n"' "\n"
    r'num ::= [0-9]+ ("." [0-9]+)? ( " "? "-" " "? [0-9]+ ("." [0-9]+)? )?' "\n"
    r'unit ::= "seconds" | "second" | "minutes" | "minute" | "mins" | "min" '
    r'| "hours" | "hour" | "hrs" | "hr" | "days" | "day" | "weeks" | "week"'
)

# Few-shot frame for gen_duration's sanity check. The MIXED Yes/No exemplars break the base model's
# reflexive-'No' default on bare "Does X take Y?" questions — which pinned even CORRECT durations at
# P(yes)~0.3 (its 'No' rationales would even recite the facts supporting Yes). With the exemplars,
# good durations land ~0.9 and bad ~0.05, so a 0.5 accept threshold works. Examples are accurate and
# don't overlap typical colony items.
DURATION_CHECK_FEWSHOT = (
    "Answer each question Yes or No.\n"
    "Question: Does watching a movie take about 2 hours?\nAnswer: Yes\n"
    "Question: Does tying your shoelaces take about 30 minutes?\nAnswer: No\n"
    "Question: Does boiling an egg take about 10 minutes?\nAnswer: Yes\n"
    "Question: Does brushing your teeth take about 4 hours?\nAnswer: No\n"
    "Question: Does {subject} take about {phrase}?\nAnswer:"
)


# ---------------------------------------------------------------------------
# llama-server client
# ---------------------------------------------------------------------------

# Ordinal degree scale for gen_percent (the 7 rungs we settled on). Each rung is scored by the
# raw sequence-logprob of the phrase as a continuation; softmax over rungs -> prob-weighted
# expected value. Phrases, not single tokens, so "a little"/"a lot" — the model's *most* common
# degree continuations — are captured properly (the probe showed " a" outranks " completely").
PERCENT_SCALE = [
    ("not at all",        0.00),
    ("barely",            0.12),
    ("a little",          0.30),
    ("moderately",        0.50),
    ("a lot",             0.70),
    ("almost completely", 0.88),
    ("completely",        1.00),
]


class LlamaServer:
    def __init__(self, url=SERVER_URL, sampling=None, seed=-1, timeout=120,
                 retries=4, retry_delay=2.0, use_grammar=True):
        self.url = url.rstrip("/")
        self.sampling = dict(DEFAULT_SAMPLING) if sampling is None else dict(sampling)
        self.seed = seed
        self.timeout = timeout
        self.retries = retries
        self.retry_delay = retry_delay
        # Grammar works on GLM-4-base + this llama.cpp build ONLY because every grammar
        # ends with a "\n" terminator and we stop on "\n" (GRAMMAR_STOP) — that halts
        # generation the instant the grammar is satisfied, BEFORE the model can sample a
        # stray special token (<|end_of_video|> etc.) that underflows the grammar stack
        # and crashes the server. Set use_grammar=False to fall back to free-gen + match.
        self.use_grammar = use_grammar

    def complete(self, prompt, grammar=None, json_schema=None, n_predict=64,
                 stop=None, temperature=None, n_probs=0, logit_bias=None,
                 cache_prompt=True, id_slot=-1):
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "cache_prompt": cache_prompt,
            "id_slot": id_slot,
            "seed": self.seed,
        }
        payload.update(self.sampling)   # explicit neutral sampler (temp 1, top_k 0, top_p 1, ...)
        if temperature is not None: payload["temperature"] = temperature
        if grammar is not None:     payload["grammar"] = grammar
        if json_schema is not None: payload["json_schema"] = json_schema
        if stop is not None:        payload["stop"] = stop
        if n_probs:                 payload["n_probs"] = n_probs
        if logit_bias is not None:  payload["logit_bias"] = logit_bias
        last = None
        for attempt in range(max(1, self.retries)):
            try:
                r = requests.post(self.url + "/completion", json=payload, timeout=self.timeout)
                r.raise_for_status()
                out = r.json()
                _emit_query_log(prompt, payload, out)
                return out
            except (requests.exceptions.RequestException, ValueError) as e:
                last = e
                time.sleep(self.retry_delay * (attempt + 1))
        raise last

    # --- free text (names, descriptions): bounded by stop + n_predict ---
    def gen_text(self, prompt, stop=None, n_predict=64, temperature=None, grammar=None):
        out = self.complete(prompt, grammar=grammar, stop=stop if stop is not None else ["\n"],
                            n_predict=n_predict, temperature=temperature)
        return clean_item(out.get("content", ""))

    # --- pick exactly one of a fixed set (replaces prsOfStrings selection) ---
    def pick_from_set(self, prompt, options, temperature=None):
        opts = [o.strip() for o in options if o.strip()]
        if not opts:
            return ""
        content = ""
        if self.use_grammar:
            # The grammar guarantees a valid option; retry a couple times if a flaky empty
            # response slips through (a 200 with empty content isn't an exception, so the
            # retry loop in complete() won't catch it).
            for _ in range(3):
                content = self.complete(prompt, grammar=alternation_grammar(opts),
                                        stop=GRAMMAR_STOP, n_predict=64,
                                        temperature=temperature).get("content", "").strip()
                if content in opts:
                    return content
                if content:
                    break   # non-empty but unmatched -> fuzzy-match below
        else:
            content = self.gen_text(prompt, stop=["\n", ".", "|", ",", ";"],
                                    n_predict=24, temperature=temperature)
        low = content.lower().strip()
        if low:                                  # NEVER fuzzy-match on empty content — that's
            for o in opts:                       # the bug that silently returned opts[0]
                if o.lower() == low:
                    return o
            for o in opts:
                if low in o.lower() or o.lower() in low:
                    return o
            rw = set(re.findall(r"\w+", low))
            best = max(opts, key=lambda o: len(rw & set(re.findall(r"\w+", o.lower()))))
            if rw & set(re.findall(r"\w+", best.lower())):
                return best
        return opts[0]   # explicit last resort (empty / no signal), not an accidental match

    # --- yes/no (hard answer) ---
    def yes_no(self, prompt, temperature=None):
        if self.use_grammar:
            out = self.complete(prompt, grammar=YES_NO_GRAMMAR, stop=GRAMMAR_STOP,
                                n_predict=5, temperature=temperature)
            return out.get("content", "").strip().lower().startswith("y")
        raw = self.gen_text(prompt, stop=["\n", ".", ","], n_predict=4, temperature=temperature)
        return raw.strip().lower().startswith("y")

    # --- yes/no probability (the old yesVsNo), done rigorously via first-token logprobs ---
    def yes_no_prob(self, prompt, n_probs=40, explain=False, explain_n=5):
        """P(yes) from the first answer token: sum the probability mass of all case/space
        variants of 'yes' vs 'no' — every token whose `.strip().lower()` is exactly 'yes' or
        'no' (' Yes', ' yes', ' YES', ... ; NOT 'yesterday'/'never'). n_predict=1; neutral
        sampling keeps the probs calibrated. (No grammar: llama.cpp reports the raw pre-grammar
        distribution in top_logprobs, so a grammar is a no-op here — the summing does the work.)
        explain=True (DEBUG ONLY) returns {p, yes, reasons_yes, reasons_no} with explain_n sampled
        rationales for BOTH sides — necessary because the No-mass is often reflexive/pedantic and one
        rationale is noise; the recurring theme across samples is the real story."""
        out = self.complete(prompt, n_predict=1, n_probs=n_probs)
        yes = no = 0.0
        for tok, p in first_token_probs(out).items():
            t = tok.strip().lower()
            if t == "yes":
                yes += p
            elif t == "no":
                no += p
        p = yes / (yes + no) if (yes + no) > 0 else 0.5
        if explain:
            return {"p": p, "yes": p >= 0.5,
                    "reasons_yes": self._reasons(prompt, "Yes", explain_n),
                    "reasons_no": self._reasons(prompt, "No", explain_n)}
        return p

    def _why(self, prompt, answer):
        """DEBUG ONLY (never on the prod path): a one-line rationale for `answer`, generated as a
        continuation of `prompt` (e.g. 'Yes, because ...'). The logprob primitives give the verdict;
        this just narrates a plausible reason for it — handy for understanding miscalibrations."""
        r = self.gen_text(prompt + f" {answer}, because", stop=["\n"], n_predict=48)
        return f"{answer}, because {r}".strip()

    def _reasons(self, prompt, answer, n=5):
        """Sample `n` independent rationales for `answer` (each a fresh `_why`). Sampling several is
        the whole point: ONE rationale is noise — one-off flukes and occasional garbage — and the REAL
        reason is the theme that recurs across samples (e.g. a reflexive 'No' whose rationales keep
        reciting the very facts that support 'Yes'). Backs the `explain=` param on every primitive."""
        return [self._why(prompt, answer) for _ in range(max(1, n))]

    # --- integer ---
    def gen_int(self, prompt, temperature=None):
        if self.use_grammar:
            raw = self.complete(prompt, grammar=INT_GRAMMAR, stop=GRAMMAR_STOP,
                               n_predict=16, temperature=temperature).get("content", "")
        else:
            raw = self.gen_text(prompt, stop=["\n", ".", ","], n_predict=12, temperature=temperature)
        m = re.search(r"-?\d+", raw)
        return int(m.group()) if m else None

    def gen_int_range(self, prompt, temperature=None):
        if self.use_grammar:
            raw = self.complete(prompt, grammar=INT_OR_RANGE_GRAMMAR, stop=GRAMMAR_STOP,
                               n_predict=24, temperature=temperature).get("content", "")
        else:
            raw = self.gen_text(prompt, stop=["\n"], n_predict=16, temperature=temperature)
        ns = [int(x) for x in re.findall(r"-?\d+", raw)]
        return (min(ns), max(ns)) if ns else None

    def gen_number(self, prompt):
        """Generate a number answer constrained by nums.NUMBER_GRAMMAR (number words OR digits,
        magnitudes, ranges) and parse it with the wholesale-copied worldcode.py parser into a
        WeightedRange (.minVal / .maxVal / .sample()). None if it fails to parse. Only the
        generation changed from the original parseNumber (token-trie + greedy/topN) -> grammar."""
        raw = self.gen_text(prompt, stop=GRAMMAR_STOP, n_predict=32,
                            grammar=nums.NUMBER_GRAMMAR if self.use_grammar else None)
        try:
            return nums.parseIntRangeFromText(raw)
        except Exception:
            return None

    def gen_number_median(self, prompt, samples=10, explain=False, explain_n=5):
        """Sample the number `samples` times (grammar-constrained) and take the MEDIAN of the
        per-sample midpoints — robust to outliers (a stray '20.000'->20, or a wild 331, won't
        skew it the way a mean would). Returns {median, lo, hi, n, samples:[{raw, value}]} or
        None if nothing parses. cache_prompt makes the repeated samples cheap. explain=True
        (DEBUG) adds `reasons`: explain_n sampled rationales for the median value."""
        out = []
        for _ in range(samples):
            raw = self.gen_text(prompt, stop=GRAMMAR_STOP, n_predict=32,
                               grammar=nums.NUMBER_GRAMMAR if self.use_grammar else None)
            try:
                rng = nums.parseIntRangeFromText(raw)
                val = (rng.minVal + rng.maxVal) / 2.0
            except Exception:
                val = None
            out.append({"raw": raw, "value": val})
        vals = [o["value"] for o in out if o["value"] is not None]
        if not vals:
            return None
        med = statistics.median(vals)
        result = {"median": med, "lo": min(vals), "hi": max(vals), "n": len(vals), "samples": out}
        if explain:
            result["reasons"] = self._reasons(prompt, f"{med:g}", explain_n)
        return result

    # --- duration -> minutes (fourth primitive): natural-unit, normalized, median over samples ---
    def gen_duration(self, prompt, samples=5, check_subject=None, check_threshold=0.5, max_rounds=3,
                     explain=False, explain_n=5):
        """Robust activity DURATION in MINUTES. A "<number> <unit>" grammar (DURATION_GRAMMAR) lets
        the model answer in whatever unit is natural — a meal in minutes, a night in hours — which is
        then normalized to minutes and median-aggregated over `samples`. This sidesteps the failure of
        forcing 'minutes' (it lowballs anything it conceives of in hours, e.g. reading 'a bed' as a nap).

        If `check_subject` is given, after each batch we SANITY-CHECK the running median with a yes/no
        'Does it make sense that {check_subject} takes about {readable}?' and RESAMPLE (pooling more
        samples, up to max_rounds) whenever P(yes) < check_threshold — so a confidently-wrong MEDIAN
        (the whole batch misreading the action, not just a stray outlier — e.g. timing cooking instead
        of eating) doesn't slip through. Pooling means each extra round only sharpens the estimate.
        Returns {minutes, lo, hi, n, rounds, p_makes_sense, samples} or None if nothing parses."""
        out, p_ok = [], None
        for rnd in range(max(1, max_rounds)):
            for _ in range(samples):
                raw = self.gen_text(prompt, stop=GRAMMAR_STOP, n_predict=16,
                                   grammar=DURATION_GRAMMAR if self.use_grammar else None)
                out.append({"raw": raw, "minutes": duration_to_minutes(raw)})
            vals = [o["minutes"] for o in out if o["minutes"] is not None]
            if not vals:
                continue                                   # nothing parsed yet — draw another batch
            med = statistics.median(vals)
            if check_subject is None:
                break                                      # no validation requested
            p_ok = self.yes_no_prob(DURATION_CHECK_FEWSHOT.format(
                subject=check_subject, phrase=minutes_to_phrase(med)))
            if p_ok >= check_threshold:
                break                                      # the model endorses this median
        vals = [o["minutes"] for o in out if o["minutes"] is not None]
        if not vals:
            return None
        med = statistics.median(vals)
        result = {"minutes": med, "lo": min(vals), "hi": max(vals),
                  "n": len(vals), "rounds": rnd + 1, "p_makes_sense": p_ok, "samples": out}
        if explain:
            result["reasons"] = self._reasons(prompt, minutes_to_phrase(med), explain_n)
        return result

    # --- verbalized magnitude -> calibrated percent (third primitive, next to yes_no_prob/number) ---
    def tokenize_pieces(self, text):
        """[(token_id, piece_str), ...] for `text` via /tokenize?with_pieces. The pieces are the
        literal substrings, so ''.join(pieces) == text — used to rebuild a growing prefix."""
        r = requests.post(self.url + "/tokenize", json={"content": text, "with_pieces": True},
                          timeout=self.timeout)
        r.raise_for_status()
        return [(t["id"], t["piece"]) for t in r.json()["tokens"]]

    @staticmethod
    def _probs_by_id(out):
        """{token_id: prob} for the first generated token, from top_logprobs (which carries the
        token id — the `probs` list does NOT, so id-lookups against it silently miss)."""
        cp = out.get("completion_probabilities") or out.get("logprobs")
        if not cp:
            return {}
        d = {}
        for c in (cp[0].get("top_logprobs") or []):
            if c.get("id") is not None and "logprob" in c:
                d[c["id"]] = math.exp(c["logprob"])
        return d

    def _phrase_logprob(self, prompt, phrase, n_probs=200, floor=1e-7):
        """Deterministic raw sequence logprob of `phrase` (already leading-spaced) as a
        continuation, via TEACHER-FORCING through the stable no-grammar reads.

        Forcing the exact string with a GBNF grammar made llama.cpp non-deterministically
        mis-report the chosen token's logprob (bimodal junk reads). The no-grammar first-token
        distribution is stable, so instead we tokenize `phrase` to its canonical tokens and sum
        logP(token_i | prompt + pieces[:i]), looking each token up BY ID in top_logprobs. A token
        below the top-`n_probs` cutoff is floored. cache_prompt keeps the growing-prefix reads cheap."""
        total, prefix = 0.0, prompt
        for tid, piece in self.tokenize_pieces(phrase):
            d = self._probs_by_id(self.complete(prefix, n_predict=1, n_probs=n_probs))
            total += math.log(max(d.get(tid, floor), 1e-12))
            prefix += piece
        return total

    def gen_percent(self, prompt, scale=None, explain=False, explain_n=5):
        """Verbalized magnitude -> calibrated [0,1]. `prompt` is framed so a degree phrase is the
        natural next words (e.g. "...satisfies their hunger"); we score each rung of the ordinal
        `scale` [(phrase, value)...] by its raw sequence-logprob, softmax over the rungs, and
        return the probability-weighted expected value. Reads the model's *ordering* of degree
        words (which it's good at) rather than asking for a percent (which clusters on round
        numbers); a single distribution read, no sampling. Returns {value, dist:[(phrase,prob,val)]}.
        explain=True (DEBUG) adds `reasons`: explain_n sampled rationales for the top-ranked phrase."""
        scale = scale or PERCENT_SCALE
        scored = []
        for ph, v in scale:
            lp = self._phrase_logprob(prompt, " " + ph)
            if lp is not None:
                scored.append((ph, v, lp))
        if not scored:
            return None
        hi = max(lp for _, _, lp in scored)
        ws = [(ph, v, math.exp(lp - hi)) for ph, v, lp in scored]
        z = sum(w for _, _, w in ws)
        value = sum(v * w for _, v, w in ws) / z
        dist = sorted(((ph, w / z, v) for ph, v, w in ws), key=lambda x: -x[1])
        result = {"value": value, "dist": dist}
        if explain:
            result["reasons"] = self._reasons(prompt, dist[0][0], explain_n)
        return result

    # --- list of items ---
    # NB: JSON-schema also compiles to a grammar and would hit the same special-token
    # crash, so we generate a newline list and split (terminator-safe via "\n\n" stop).
    def gen_list(self, prompt, n, temperature=None):
        out = self.complete(prompt, stop=["\n\n"], n_predict=40 * n, temperature=temperature)
        items = [clean_item(re.sub(r"^[-*\d.\)\s]+", "", x)) for x in out.get("content", "").split("\n")]
        return [x for x in items if x][:n]


# ---------------------------------------------------------------------------
# small text utilities (trimmed from worldcode.py — most cleanup is now unneeded)
# ---------------------------------------------------------------------------

def clean_item(item: str) -> str:
    item = item.split("<|")[0]                  # cut any leaked special token (<|end_of_text|> ...)
    item = re.sub(r"\s+", " ", item.strip())
    item = re.sub(r"[.]$", "", item).strip()
    return item


_DUR_UNIT_MIN = {"sec": 1.0 / 60, "min": 1.0, "hour": 60.0, "hr": 60.0,
                 "day": 1440.0, "week": 10080.0}


def duration_to_minutes(text):
    """Parse '<number>[-<number>] <unit>' (' 8 hours', '20-30 min', '90 seconds') -> minutes; a
    range is averaged. Returns None if no number+unit is present. Backs gen_duration."""
    if not text:
        return None
    t = text.strip().lower().replace("–", "-")     # en-dash -> hyphen
    m = re.search(r"(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*"
                  r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)", t)
    if not m:
        return None
    a = float(m.group(1))
    b = float(m.group(2)) if m.group(2) else a
    u = m.group(3).rstrip("s")                          # hours->hour, mins->min, secs->sec, hrs->hr
    per = _DUR_UNIT_MIN.get(u) or _DUR_UNIT_MIN.get(u[:3])
    return (a + b) / 2.0 * per if per else None


def minutes_to_phrase(m):
    """Human-readable duration for the sanity-check question (480 -> '8 hours', 1.5 -> '2 minutes')."""
    if m < 1:
        return "less than a minute"
    if m < 90:
        return f"{round(m)} minutes"
    if m < 1440:
        h = m / 60.0
        return f"{int(h)} hours" if abs(h - round(h)) < 1e-9 else f"{round(h, 1)} hours"
    d = m / 1440.0
    return f"{int(d)} days" if abs(d - round(d)) < 1e-9 else f"{round(d, 1)} days"


def first_token_probs(completion_json) -> dict:
    """{token_str: prob} for the first generated token, defensive across schemas."""
    cp = completion_json.get("completion_probabilities") or completion_json.get("logprobs")
    if not cp:
        return {}
    cand = cp[0].get("probs") or cp[0].get("top_logprobs") or []
    out = {}
    for c in cand:
        tok = c.get("tok") or c.get("token") or c.get("text") or ""
        if "prob" in c:
            out[tok] = c["prob"]
        elif "logprob" in c:
            out[tok] = math.exp(c["logprob"])
    return out


def embed_texts(texts, url=EMBED_URL, timeout=30):
    """Embed texts via the local embeddinggemma server (OpenAI-style /v1/embeddings).
    Uses embeddinggemma's document prompt format. Returns a list of vectors."""
    inp = [f"title: none | text: {t}" for t in texts]
    r = requests.post(url.rstrip("/") + "/v1/embeddings", json={"input": inp}, timeout=timeout)
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def iter_unique(server, prompt, n=12, max_attempts=None, grammar=None, stop=None,
                sim_threshold=0.78, max_len=40, strip_articles=True, reject=None,
                seed=None, embed_url=EMBED_URL):
    """Generate up to `n` unique short items by RE-SAMPLING a fixed `prompt` one line at a time
    (warm temp) and semantic-deduping with embeddinggemma — an item is skipped if its cosine to
    any kept item is >= sim_threshold. Yields each kept item. The shared engine behind
    iter_locations / iter_objects / need generation. `seed` pre-loads the dedup set (so e.g.
    generated needs don't repeat the universal core); `reject(item)->bool` drops unwanted items
    (non-English, multi-word, ...); `strip_articles` removes a leading the/a/an."""
    embs, seen, count = [], set(), 0
    if seed:
        for sd in seed:
            seen.add(sd.strip().lower())
            try:
                embs.append(embed_texts([sd], embed_url)[0])
            except Exception:
                pass
    for _ in range(max_attempts or n * 5):
        if count >= n:
            break
        raw = server.gen_text(prompt, stop=stop or ["\n"], n_predict=16, grammar=grammar)
        item = clean_item(re.sub(r"\s*\(.*?\)", "", raw))
        words = item.split()
        if strip_articles and words and words[0].lower() in ("the", "a", "an"):
            item = " ".join(words[1:])
        if not item or len(item) > max_len or item.lower() in seen or (reject and reject(item)):
            continue
        try:
            e = embed_texts([item], embed_url)[0]
            if embs and max(cosine(e, pe) for pe in embs) >= sim_threshold:
                continue
        except Exception:
            e = None
        seen.add(item.lower())
        if e is not None:
            embs.append(e)
        count += 1
        yield item
