# Factoring a trait hierarchy out of the flat affordance matrix

**Question.** We're planning to pre-bake a flat table: for every (species, item) pair, which needs it
fills, plus for every species which needs it has. Is there an algorithm that automatically extracts a
tree (or DAG) of *traits* from that data — so that "most things inherit from `food`, fill in only the
diffs" falls out rather than being hand-imposed?

**Short answer.** Yes. It's a lovely inversion of the whole problem: instead of *imposing* a taxonomy and
inheriting down it, you bake the flat matrix and let the hierarchy **fall out of the data**. This is
unsupervised structure learning from a binary object×attribute matrix, and there are a few mature
algorithm families — each a slightly different answer to "what is a trait."

---

## The data

A binary (or weighted) co-occurrence matrix:

- **rows** = items (and, separately, species)
- **columns** = needs / affordances (fills-hunger, provides-warmth, satisfies-social, …)
- **cell** = does this item fill this need (1/0, or the refill magnitude)

Plus the species→needs matrix (which needs a species *has*), and the species×species predator/prey/social
matrix — all the same shape. A "trait" is latent structure in this matrix: a bundle of affordances that a
group of rows shares (e.g. everything that fills hunger → a `food` trait).

---

## 1. The exact match — Formal Concept Analysis (FCA)

This is almost literally the question. Feed FCA a *formal context* (items × needs, 1 = fills) and it
computes the **concept lattice**. Every *concept* is a maximal set of items sharing a maximal set of
needs:

- the shared-need set **is the trait** (`{fills hunger}` → the food concept),
- the lattice orders concepts **general → specific** (object ⊃ food ⊃ bread).

It's the cleanest formalization of "extract traits *and* their hierarchy from a co-occurrence matrix."

**Caveats**

- It yields a **lattice (DAG), not a tree** — which is actually *correct*: a roast chicken is food *and*
  warm = two traits, i.e. multiple inheritance.
- The number of concepts can blow up combinatorially, so you prune: keep only frequent concepts
  ("iceberg lattices") or rank by concept *stability*.

## 2. The simple practical one — hierarchical clustering

Cluster items by their need-profile (Jaccard / Hamming on binary rows, cosine on weighted), get a
dendrogram, and label each cluster by the needs its members share.

- Fast, always a **tree**.
- But traits are *implicit* — you read them off the clusters post-hoc rather than getting them as
  first-class objects the way FCA does.

Good for a first eyeball of "are there clean clusters here at all."

## 3. The most apt reframing — phylogenetics

Your matrix *is* a cladistic **character matrix**: items/species are taxa, needs are characters.

- **Neighbor-joining** or **maximum parsimony** will infer a trait-tree.
- A "shared derived trait that defines a clade" (a *synapomorphy*) is exactly "the trait all foods share."
- Parsimony's inductive bias — prefer the tree where traits are gained/lost the **fewest** times — is a
  genuinely principled, inheritance-friendly objective (Occam over trait-changes).

It assumes a strict tree (single inheritance) and an evolutionary-descent model that isn't literally true
here, but as a *clustering* objective the parsimony bias is well-matched to "the most inheritance-efficient
arrangement."

## 4. The deep framing — MDL (and why *learned* beats *imposed*)

All of the above are really approximating one thing: the hierarchy that lets you **describe the affordance
matrix in the fewest bits** — traits up top as reusable patterns, exceptions stored as overrides. That *is*
our "inherit defaults, store only diffs," and **Minimum Description Length** makes "how good is this tree"
precise:

> quality of a hierarchy = size(shared trait tables) + size(per-item diffs)

Algorithms that optimize this directly: MDL pattern-set mining (**Krimp / Slim**) and MDL **Boolean matrix
factorization** (each latent factor = a co-occurring affordance bundle = a trait).

The payoff, and the strongest argument for the data-driven route over plugging in WordNet/SUMO:

> A **data-extracted trait-tree is inheritance-optimal by construction.** An imposed ontology can't
> guarantee its categories actually predict affordances; a learned one is *defined* by them.

The whole point of the hierarchy was to compress the affordance table — so extracting it *from* that table
directly optimizes the thing we care about.

---

## The one recurring decision: tree vs DAG

| | strict tree (clustering / phylogeny) | lattice / DAG (FCA) |
|---|---|---|
| multiple inheritance (food **and** warm) | ✗ lost | ✓ preserved |
| simplicity | ✓ simpler | ✗ heavier |
| faithfulness to the data | lossy | faithful |

This is the same tree-vs-DAG choice that keeps surfacing. Most-specific-wins + item-level overrides makes a
tree workable; a DAG is more correct but you pay for it.

---

## How it composes with the flat-bake plan

The two ideas snap together:

1. **Bake the flat matrix** (a wide family of items × species × needs) — the experiment we're starting with.
2. **Run FCA (or MDL-clustering) on it** → the extracted trait-DAG *is* the pack hierarchy, derived rather
   than guessed.
3. It also answers, empirically and *before* committing to any machinery, **whether clean traits even
   exist** and **how much they'd compress the matrix** — i.e. whether the hierarchy pays for itself.

**Concrete next step.** Once the flat table is baked, run FCA and an MDL-clustering over it and report the
extracted traits + the achieved compression (shared-tables + diffs vs the raw matrix). That's a cheap,
concrete read on real numbers — if compression is high and the traits look sensible, the hierarchy is worth
building; if not, the flat table (+ embedding-nearest-neighbour for fuzzy coverage) may be enough on its own.

---

## Pointers

- **FCA** — Ganter & Wille, *Formal Concept Analysis*; iceberg lattices (Stumme), concept stability (Kuznetsov).
- **Hierarchical clustering** — agglomerative linkage + dendrograms; Jaccard distance for binary profiles.
- **Phylogenetics** — neighbor-joining (Saitou & Nei); maximum parsimony; synapomorphy / cladistics.
- **MDL structure learning** — Krimp / Slim (Vreeken, van Leeuwen, Siebes); MDL Boolean matrix factorization
  (Miettinen & Vreeken).
