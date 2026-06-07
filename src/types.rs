use vstd::prelude::*;

verus! {

///  A term in a rule pattern: either a literal value or a variable to be bound.
pub enum Term {
    Lit(int),
    Var(nat),
}

///  Comparison operators for guard conditions.
pub enum CmpOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

///  A triple pattern used in rule conditions and effects: (entity, key, value).
pub struct TriplePattern {
    pub entity: Term,
    pub key: Term,
    pub value: Term,
}

///  A guard: compare two terms with a comparison operator.
pub struct Guard {
    pub left: Term,
    pub op: CmpOp,
    pub right: Term,
}

///  A rule with monotonic id, pattern conditions, guards, and effects.
pub struct Rule {
    pub id: nat,
    pub conditions: Seq<TriplePattern>,
    pub guards: Seq<Guard>,
    pub effects: Seq<TriplePattern>,
}

///  A binding maps variable ids to concrete values.
pub type Binding = Map<nat, int>;

///  Structural equality on terms.
pub open spec fn term_eq(a: Term, b: Term) -> bool {
    match a {
        Term::Lit(x) => match b {
            Term::Lit(y) => x == y,
            Term::Var(_) => false,
        },
        Term::Var(x) => match b {
            Term::Lit(_) => false,
            Term::Var(y) => x == y,
        },
    }
}

///  Structural equality on triple patterns.
pub open spec fn triple_pattern_eq(a: TriplePattern, b: TriplePattern) -> bool {
    term_eq(a.entity, b.entity)
    && term_eq(a.key, b.key)
    && term_eq(a.value, b.value)
}

///  term_eq is reflexive.
pub proof fn lemma_term_eq_reflexive(t: Term)
    ensures term_eq(t, t),
{
}

///  term_eq is symmetric.
pub proof fn lemma_term_eq_symmetric(a: Term, b: Term)
    ensures term_eq(a, b) == term_eq(b, a),
{
}

///  triple_pattern_eq is reflexive.
pub proof fn lemma_triple_pattern_eq_reflexive(t: TriplePattern)
    ensures triple_pattern_eq(t, t),
{
    lemma_term_eq_reflexive(t.entity);
    lemma_term_eq_reflexive(t.key);
    lemma_term_eq_reflexive(t.value);
}

///  triple_pattern_eq is symmetric.
pub proof fn lemma_triple_pattern_eq_symmetric(a: TriplePattern, b: TriplePattern)
    ensures triple_pattern_eq(a, b) == triple_pattern_eq(b, a),
{
    lemma_term_eq_symmetric(a.entity, b.entity);
    lemma_term_eq_symmetric(a.key, b.key);
    lemma_term_eq_symmetric(a.value, b.value);
}

///  Substitute a term under a binding: variables are replaced, literals pass through.
pub open spec fn substitute_term(t: Term, binding: Binding) -> int {
    match t {
        Term::Lit(v) => v,
        Term::Var(id) => if binding.contains_key(id) { binding[id] } else { 0 },
    }
}

///  Evaluate a comparison operator on two integer values.
pub open spec fn eval_cmp(op: CmpOp, left: int, right: int) -> bool {
    match op {
        CmpOp::Eq => left == right,
        CmpOp::Ne => left != right,
        CmpOp::Lt => left < right,
        CmpOp::Le => left <= right,
        CmpOp::Gt => left > right,
        CmpOp::Ge => left >= right,
    }
}

///  Evaluate a guard under a binding.
pub open spec fn eval_guard(g: Guard, binding: Binding) -> bool {
    eval_cmp(g.op, substitute_term(g.left, binding), substitute_term(g.right, binding))
}

///  All guards in a sequence hold under a binding.
pub open spec fn all_guards_hold(guards: Seq<Guard>, binding: Binding) -> bool {
    forall|i: int| 0 <= i < guards.len() ==> eval_guard(#[trigger] guards[i], binding)
}

} //  verus!
