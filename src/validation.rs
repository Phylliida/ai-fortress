use vstd::prelude::*;
use crate::types::*;

verus! {

///  Whether every triple pattern in `effects` appears in `conditions` (syntactic subset).
///  This is the "trivial rule" check from the pseudocode: effects ⊆ conditions.
pub open spec fn effects_subset_of_conditions(rule: Rule) -> bool {
    forall|i: int| #![trigger rule.effects[i]]
        0 <= i < rule.effects.len() ==>
        exists|j: int| #![trigger rule.conditions[j]]
            0 <= j < rule.conditions.len()
            && triple_pattern_eq(rule.effects[i], rule.conditions[j])
}

///  Whether a specific effect index has no matching condition.
pub open spec fn is_novel_effect_at(rule: Rule, i: int) -> bool {
    0 <= i < rule.effects.len()
    && forall|j: int| #![trigger rule.conditions[j]]
        0 <= j < rule.conditions.len()
        ==> !triple_pattern_eq(rule.effects[i], rule.conditions[j])
}

///  A rule is valid iff it has at least one novel effect.
///  From pseudocode: validate(rule) → Bool: return rule.effects ⊄ rule.conditions
pub open spec fn validate_rule(rule: Rule) -> bool {
    exists|i: int| #![trigger rule.effects[i]] is_novel_effect_at(rule, i)
}

//  --- Lemmas ---

///  A rule with empty effects has no novel effect (and thus is invalid).
pub proof fn lemma_empty_effects_invalid(rule: Rule)
    requires rule.effects.len() == 0
    ensures !validate_rule(rule),
{
}

///  A rule with non-empty effects and empty conditions is always valid.
pub proof fn lemma_nonempty_effects_empty_conditions_valid(rule: Rule)
    requires
        rule.effects.len() > 0,
        rule.conditions.len() == 0,
    ensures validate_rule(rule),
{
    //  Explicitly prove the body of the existential for witness i = 0
    assert(is_novel_effect_at(rule, 0int));
    //  Access rule.effects[0] to fire the existential trigger
    let _ = spec_affirm(rule.effects[0int]);
}

///  If a specific effect index has no matching condition, the rule is valid.
pub proof fn lemma_novel_effect_at_implies_valid(rule: Rule, idx: int)
    requires is_novel_effect_at(rule, idx)
    ensures validate_rule(rule),
{
    assert(is_novel_effect_at(rule, idx));
    let _ = spec_affirm(rule.effects[idx]);
}

//  Helper to force a trigger to fire without side effects.
#[verifier::inline]
pub open spec fn spec_affirm<T>(x: T) -> T { x }

} //  verus!
