use vstd::prelude::*;

verus! {

///  Priority for conflict resolution.
///  Higher specificity (more conditions + guards) wins.
///  Ties broken by higher rule id (newer rule wins).
pub struct Priority {
    pub specificity: int,
    pub rule_id: nat,
}

///  Compute priority from a rule's condition and guard counts.
pub open spec fn compute_priority(num_conditions: int, num_guards: int, rule_id: nat) -> Priority {
    Priority {
        specificity: num_conditions + num_guards,
        rule_id: rule_id,
    }
}

///  Strict "greater than" on priorities (lexicographic).
pub open spec fn priority_gt(a: Priority, b: Priority) -> bool {
    a.specificity > b.specificity
    || (a.specificity == b.specificity && a.rule_id > b.rule_id)
}

///  Equality on priorities.
pub open spec fn priority_eq(a: Priority, b: Priority) -> bool {
    a.specificity == b.specificity && a.rule_id == b.rule_id
}

///  Greater-than-or-equal on priorities.
pub open spec fn priority_ge(a: Priority, b: Priority) -> bool {
    priority_gt(a, b) || priority_eq(a, b)
}

//  --- Total order proofs ---

///  priority_gt is irreflexive.
pub proof fn lemma_priority_gt_irreflexive(a: Priority)
    ensures !priority_gt(a, a),
{
}

///  priority_gt is asymmetric.
pub proof fn lemma_priority_gt_asymmetric(a: Priority, b: Priority)
    ensures priority_gt(a, b) ==> !priority_gt(b, a),
{
}

///  priority_gt is transitive.
pub proof fn lemma_priority_gt_transitive(a: Priority, b: Priority, c: Priority)
    ensures priority_gt(a, b) && priority_gt(b, c) ==> priority_gt(a, c),
{
}

///  Trichotomy: exactly one of gt, lt, or eq holds.
pub proof fn lemma_priority_trichotomy(a: Priority, b: Priority)
    ensures
        priority_gt(a, b) || priority_gt(b, a) || priority_eq(a, b),
        !(priority_gt(a, b) && priority_gt(b, a)),
        !(priority_gt(a, b) && priority_eq(a, b)),
        !(priority_gt(b, a) && priority_eq(a, b)),
{
}

///  priority_ge is reflexive.
pub proof fn lemma_priority_ge_reflexive(a: Priority)
    ensures priority_ge(a, a),
{
}

///  priority_ge is transitive.
pub proof fn lemma_priority_ge_transitive(a: Priority, b: Priority, c: Priority)
    ensures priority_ge(a, b) && priority_ge(b, c) ==> priority_ge(a, c),
{
}

///  priority_ge is antisymmetric (up to priority_eq).
pub proof fn lemma_priority_ge_antisymmetric(a: Priority, b: Priority)
    ensures priority_ge(a, b) && priority_ge(b, a) ==> priority_eq(a, b),
{
}

//  --- Derived properties matching pseudocode ---

///  More conditions/guards always means higher priority (regardless of id).
pub proof fn lemma_more_specific_wins(a: Priority, b: Priority)
    requires a.specificity > b.specificity
    ensures priority_gt(a, b),
{
}

///  Equal specificity: newer rule (higher id) wins.
pub proof fn lemma_newer_wins_ties(a: Priority, b: Priority)
    requires
        a.specificity == b.specificity,
        a.rule_id > b.rule_id,
    ensures priority_gt(a, b),
{
}

///  !priority_ge(a, b) implies priority_gt(b, a).
pub proof fn lemma_not_ge_implies_gt_reverse(a: Priority, b: Priority)
    ensures !priority_ge(a, b) ==> priority_gt(b, a),
{
}

} //  verus!
