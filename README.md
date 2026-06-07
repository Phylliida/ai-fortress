# bootstrapped-llm-sim
Formally verified LLM-bootstrapped rule simulation engine.

## Verification

Requires [Verus](https://github.com/verus-lang/verus) built in `../verus/`.

```bash
./scripts/check.sh --require-verus --forbid-trusted-escapes
```

## Design

```
//  =====================================================
//   LLM-BOOTSTRAPPED SIMULATION ENGINE  (minimal)
//  =====================================================
//
//   NOTE ON RULE ACCUMULATION
//   -------------------------
//   Rules are never pruned. Matching is an indexed lookup, so
//   millions of inert rules cost storage but not time. Rules that
//   produce wrong outputs are naturally overridden by more-specific
//   rules the LLM generates later (conflict resolution handles this).
//
//   The one degenerate case is two contradictory rules at equal
//   specificity, which causes oscillation in the fixed-point loop.
//   This is handled by the tiebreaker in conflict resolution:
//   newer rules win ties, since they were generated with more
//   recent world context.
//

//  --- Store ---

store  : Map<(Entity, Key), Value>
active : Set<Entity>

set(e, k, v) → changed:Bool:
  if store[e,k] == v: return false
  store[e,k] = v
  return true

get(e, k) → Value?
has(e, k) → Bool
all(k)    → [(Entity, Value)]:
  return [(e, v) for (e2, k2), v in store if k2 == k]


//  --- Rules ---

Rule:
  id         : Int                        //  monotonic, used as tiebreaker
  conditions : [(term, term, term)]
  guards     : [(term, cmp, term)]
  effects    : [(term, term, term)]

rules : List<Rule>
next_id : Int = 0

match(rule) → [Binding]:
  bindings = [∅]
  for (e, k, v) in rule.conditions:
    bindings = [b ∪ unify(e, k, v, triple, b)
                for b in bindings
                for triple in (all(k) if k is literal else store)
                if compatible(b, triple)]
  return [b for b in bindings if all guards hold]


//  --- Evaluate ---
//  Most-specific wins. Ties: highest id (newest rule).
//  Runs to fixed point within a tick.

priority(rule) → (Int, Int):
  return (len(rule.conditions) + len(rule.guards), rule.id)

evaluate(rules) → Set<Entity>:
  touched = {}

  loop:
    mutations = Map<(E,K), (Value, (Int,Int))>
    fired = false

    for rule in rules:
      pri = priority(rule)
      for binding in match(rule):
        for effect in rule.effects:
          (e, k, v) = substitute(effect, binding)
          if (e,k) ∉ mutations or pri > mutations[e,k].pri:
            mutations[e,k] = (v, pri)

    for (e,k), (v, _) in mutations:
      if set(e, k, v):
        fired = true
        touched.add(e)

    if not fired or iterations > MAX_ITER:
      break

  return touched


//  --- Miss Detection ---
//  Reactive entity that nothing touched → miss.
//  Simple cooldown prevents repeated LLM calls.

cooldowns : Map<Entity, Int>

detect(touched) → [Context]:
  misses = []
  for e in active:
    if get(e, "reactive") and e ∉ touched and cooldowns[e] ≤ 0:
      misses.add(context_for(e))
      cooldowns[e] = COOLDOWN_TICKS
  return misses

context_for(e) → Context:
  return { all triples for e } ∪ { triples for nearby entities }


//  --- LLM ---

request_rules(contexts):
  response = llm(format(contexts))
  for rule in parse(response):
    if validate(rule):
      rule.id = next_id++
      rules.add(rule)

validate(rule) → Bool:
  return rule.effects ⊄ rule.conditions


//  --- Main Loop ---

tick(rules):
  for e in cooldowns: cooldowns[e] -= 1
  touched = evaluate(rules)
  misses  = detect(touched)
  if misses not empty:
    request_rules(misses)
```
