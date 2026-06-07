use vstd::prelude::*;
use crate::priority::*;

verus! {

///  A pending mutation: proposed change to (entity, key) with a value and priority.
pub struct Mutation {
    pub entity: int,
    pub key: int,
    pub value: int,
    pub pri: Priority,
}

///  Whether a mutation targets a given (entity, key) pair.
pub open spec fn mutation_targets(m: Mutation, entity: int, key: int) -> bool {
    m.entity == entity && m.key == key
}

///  Resolve mutations: for a given (entity, key), find the winning mutation
///  (highest priority). Returns None if no mutation targets that pair.
pub open spec fn resolve(
    mutations: Seq<Mutation>,
    entity: int,
    key: int,
) -> Option<Mutation>
    decreases mutations.len(),
{
    if mutations.len() == 0 {
        None
    } else {
        let last = mutations.last();
        let rest = resolve(mutations.drop_last(), entity, key);
        if !mutation_targets(last, entity, key) {
            rest
        } else {
            match rest {
                None => Some(last),
                Some(prev) => {
                    if priority_ge(last.pri, prev.pri) {
                        Some(last)
                    } else {
                        Some(prev)
                    }
                },
            }
        }
    }
}

///  Whether any mutation in the sequence targets (entity, key).
pub open spec fn any_targets(mutations: Seq<Mutation>, entity: int, key: int) -> bool {
    exists|i: int| 0 <= i < mutations.len() && mutation_targets(#[trigger] mutations[i], entity, key)
}

//  --- Lemmas ---

///  resolve of an empty sequence is None.
pub proof fn lemma_resolve_empty(entity: int, key: int)
    ensures resolve(Seq::<Mutation>::empty(), entity, key) is None,
{
}

///  If no mutation targets (e, k), resolve returns None.
pub proof fn lemma_resolve_none_if_no_targets(
    mutations: Seq<Mutation>,
    entity: int,
    key: int,
)
    requires !any_targets(mutations, entity, key)
    ensures resolve(mutations, entity, key) is None
    decreases mutations.len(),
{
    if mutations.len() > 0 {
        lemma_resolve_none_if_no_targets(mutations.drop_last(), entity, key);
    }
}

///  If some mutation targets (e, k), resolve returns Some.
pub proof fn lemma_resolve_some_if_targets(
    mutations: Seq<Mutation>,
    entity: int,
    key: int,
)
    requires any_targets(mutations, entity, key)
    ensures resolve(mutations, entity, key) is Some
    decreases mutations.len(),
{
    let i = choose|i: int| 0 <= i < mutations.len() && mutation_targets(#[trigger] mutations[i], entity, key);
    if mutations.len() > 0 {
        let last_idx = (mutations.len() - 1) as int;
        if mutation_targets(mutations[last_idx], entity, key) {
            //  last element targets (e,k), so resolve returns Some
        } else {
            //  i must be in drop_last
            assert(0 <= i < mutations.drop_last().len());
            assert(mutations.drop_last()[i] == mutations[i]);
            lemma_resolve_some_if_targets(mutations.drop_last(), entity, key);
        }
    }
}

///  The resolved mutation always targets the queried (entity, key).
pub proof fn lemma_resolve_targets(
    mutations: Seq<Mutation>,
    entity: int,
    key: int,
)
    requires resolve(mutations, entity, key) is Some
    ensures mutation_targets(resolve(mutations, entity, key).unwrap(), entity, key)
    decreases mutations.len(),
{
    if mutations.len() > 0 {
        let last = mutations.last();
        let rest = resolve(mutations.drop_last(), entity, key);
        if !mutation_targets(last, entity, key) {
            lemma_resolve_targets(mutations.drop_last(), entity, key);
        } else {
            match rest {
                None => {},
                Some(prev) => {
                    lemma_resolve_targets(mutations.drop_last(), entity, key);
                },
            }
        }
    }
}

///  A single targeting mutation is returned by resolve.
pub proof fn lemma_resolve_singleton(m: Mutation, entity: int, key: int)
    requires mutation_targets(m, entity, key)
    ensures
        resolve(seq![m], entity, key) is Some,
        ({
            let result = resolve(seq![m], entity, key).unwrap();
            result.entity == m.entity
            && result.key == m.key
            && result.value == m.value
            && priority_eq(result.pri, m.pri)
        }),
{
    let s = seq![m];
    assert(s.len() == 1);
    assert(s.drop_last() =~= Seq::<Mutation>::empty());
    assert(resolve(s.drop_last(), entity, key) is None);
}

} //  verus!
