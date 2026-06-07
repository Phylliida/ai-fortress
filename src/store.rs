use vstd::prelude::*;

verus! {

///  The simulation store: maps (entity, key) pairs to values,
///  with an active entity set.
pub struct Store {
    pub data: Map<(int, int), int>,
    pub active: Set<int>,
}

///  Set a value in the store. Returns (new_store, changed).
pub open spec fn store_set(s: Store, entity: int, key: int, value: int) -> (Store, bool) {
    let changed = !s.data.contains_key((entity, key)) || s.data[(entity, key)] != value;
    let new_data = s.data.insert((entity, key), value);
    (Store { data: new_data, active: s.active }, changed)
}

///  Get a value from the store.
pub open spec fn store_get(s: Store, entity: int, key: int) -> Option<int> {
    if s.data.contains_key((entity, key)) {
        Some(s.data[(entity, key)])
    } else {
        None
    }
}

///  Check if a key exists for an entity.
pub open spec fn store_has(s: Store, entity: int, key: int) -> bool {
    s.data.contains_key((entity, key))
}

///  Collect all (entity, value) pairs for a given key.
pub open spec fn store_all(s: Store, key: int) -> Set<(int, int)> {
    Set::new(|pair: (int, int)| s.data.contains_key((pair.0, key)) && s.data[(pair.0, key)] == pair.1)
}

//  --- Lemmas ---

///  After set, the value is stored.
pub proof fn lemma_store_set_stores_value(s: Store, entity: int, key: int, value: int)
    ensures
        store_set(s, entity, key, value).0.data.contains_key((entity, key)),
        store_set(s, entity, key, value).0.data[(entity, key)] == value,
{
}

///  The changed flag is true iff the value actually changed.
pub proof fn lemma_store_set_changed_correct(s: Store, entity: int, key: int, value: int)
    ensures
        store_set(s, entity, key, value).1 == (
            !s.data.contains_key((entity, key)) || s.data[(entity, key)] != value
        ),
{
}

///  Setting a value preserves other entries.
pub proof fn lemma_store_set_preserves_others(
    s: Store, entity: int, key: int, value: int,
    e2: int, k2: int,
)
    requires (e2, k2) != (entity, key)
    ensures
        store_set(s, entity, key, value).0.data.contains_key((e2, k2))
            == s.data.contains_key((e2, k2)),
        s.data.contains_key((e2, k2)) ==>
            store_set(s, entity, key, value).0.data[(e2, k2)] == s.data[(e2, k2)],
{
}

///  Setting a value does not change the active set.
pub proof fn lemma_store_set_preserves_active(s: Store, entity: int, key: int, value: int)
    ensures store_set(s, entity, key, value).0.active == s.active,
{
}

///  Setting the same value yields changed == false.
pub proof fn lemma_store_set_idempotent(s: Store, entity: int, key: int, value: int)
    requires
        s.data.contains_key((entity, key)),
        s.data[(entity, key)] == value,
    ensures
        !store_set(s, entity, key, value).1,
{
}

///  get after set on the same key returns the set value.
pub proof fn lemma_get_after_set_same(s: Store, entity: int, key: int, value: int)
    ensures
        store_get(store_set(s, entity, key, value).0, entity, key) == Some(value),
{
}

///  get after set on a different key is unchanged.
pub proof fn lemma_get_after_set_other(
    s: Store, entity: int, key: int, value: int,
    e2: int, k2: int,
)
    requires (e2, k2) != (entity, key)
    ensures
        store_get(store_set(s, entity, key, value).0, e2, k2) == store_get(s, e2, k2),
{
}

///  has after set on the same key is true.
pub proof fn lemma_has_after_set(s: Store, entity: int, key: int, value: int)
    ensures
        store_has(store_set(s, entity, key, value).0, entity, key),
{
}

} //  verus!
