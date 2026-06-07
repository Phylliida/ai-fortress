use vstd::prelude::*;
use crate::store::*;

verus! {

///  Sentinel key representing the "reactive" attribute in the store.
pub open spec fn REACTIVE_KEY() -> int { 0 }

///  Default cooldown ticks after a miss is detected.
pub open spec fn COOLDOWN_TICKS() -> int { 5 }

///  Whether an entity is reactive (has the reactive key set in the store).
pub open spec fn is_reactive(store: Store, entity: int) -> bool {
    store_has(store, entity, REACTIVE_KEY())
}

///  Whether an entity's cooldown has expired.
pub open spec fn cooldown_expired(cooldowns: Map<int, int>, entity: int) -> bool {
    !cooldowns.contains_key(entity) || cooldowns[entity] <= 0
}

///  An entity is a miss if it's active, reactive, not touched, and off cooldown.
pub open spec fn is_miss(
    store: Store,
    active: Set<int>,
    touched: Set<int>,
    cooldowns: Map<int, int>,
    entity: int,
) -> bool {
    active.contains(entity)
    && is_reactive(store, entity)
    && !touched.contains(entity)
    && cooldown_expired(cooldowns, entity)
}

///  Decrement all cooldowns by 1.
pub open spec fn tick_cooldowns(cooldowns: Map<int, int>) -> Map<int, int> {
    Map::new(
        |e: int| cooldowns.contains_key(e),
        |e: int| cooldowns[e] - 1,
    )
}

///  Set cooldown for a specific entity (after miss is detected).
pub open spec fn set_cooldown(cooldowns: Map<int, int>, entity: int) -> Map<int, int> {
    cooldowns.insert(entity, COOLDOWN_TICKS())
}

//  --- Lemmas ---

///  Soundness: if is_miss holds, all four conditions hold individually.
pub proof fn lemma_miss_sound_active(
    store: Store, active: Set<int>, touched: Set<int>,
    cooldowns: Map<int, int>, entity: int,
)
    requires is_miss(store, active, touched, cooldowns, entity)
    ensures active.contains(entity),
{
}

pub proof fn lemma_miss_sound_reactive(
    store: Store, active: Set<int>, touched: Set<int>,
    cooldowns: Map<int, int>, entity: int,
)
    requires is_miss(store, active, touched, cooldowns, entity)
    ensures is_reactive(store, entity),
{
}

pub proof fn lemma_miss_sound_untouched(
    store: Store, active: Set<int>, touched: Set<int>,
    cooldowns: Map<int, int>, entity: int,
)
    requires is_miss(store, active, touched, cooldowns, entity)
    ensures !touched.contains(entity),
{
}

pub proof fn lemma_miss_sound_cooldown(
    store: Store, active: Set<int>, touched: Set<int>,
    cooldowns: Map<int, int>, entity: int,
)
    requires is_miss(store, active, touched, cooldowns, entity)
    ensures cooldown_expired(cooldowns, entity),
{
}

///  Completeness: if all four conditions hold, is_miss holds.
pub proof fn lemma_miss_complete(
    store: Store, active: Set<int>, touched: Set<int>,
    cooldowns: Map<int, int>, entity: int,
)
    requires
        active.contains(entity),
        is_reactive(store, entity),
        !touched.contains(entity),
        cooldown_expired(cooldowns, entity),
    ensures is_miss(store, active, touched, cooldowns, entity),
{
}

///  After tick_cooldowns, all values are decremented by 1.
pub proof fn lemma_tick_cooldowns_decrements(cooldowns: Map<int, int>, entity: int)
    requires cooldowns.contains_key(entity)
    ensures
        tick_cooldowns(cooldowns).contains_key(entity),
        tick_cooldowns(cooldowns)[entity] == cooldowns[entity] - 1,
{
}

///  After tick_cooldowns, domain is preserved.
pub proof fn lemma_tick_cooldowns_preserves_domain(cooldowns: Map<int, int>, entity: int)
    ensures
        tick_cooldowns(cooldowns).contains_key(entity) == cooldowns.contains_key(entity),
{
}

///  An entity at cooldown 1 becomes expired after tick.
pub proof fn lemma_tick_expires_at_one(cooldowns: Map<int, int>, entity: int)
    requires
        cooldowns.contains_key(entity),
        cooldowns[entity] == 1,
    ensures cooldown_expired(tick_cooldowns(cooldowns), entity),
{
}

///  An entity at cooldown > 1 is still cooling down after tick.
pub proof fn lemma_tick_still_cooling(cooldowns: Map<int, int>, entity: int)
    requires
        cooldowns.contains_key(entity),
        cooldowns[entity] > 1,
    ensures !cooldown_expired(tick_cooldowns(cooldowns), entity),
{
}

///  After set_cooldown, entity has COOLDOWN_TICKS remaining.
pub proof fn lemma_set_cooldown_value(cooldowns: Map<int, int>, entity: int)
    ensures
        set_cooldown(cooldowns, entity).contains_key(entity),
        set_cooldown(cooldowns, entity)[entity] == COOLDOWN_TICKS(),
{
}

///  After set_cooldown, entity is not expired.
pub proof fn lemma_set_cooldown_not_expired(cooldowns: Map<int, int>, entity: int)
    ensures !cooldown_expired(set_cooldown(cooldowns, entity), entity),
{
}

} //  verus!
