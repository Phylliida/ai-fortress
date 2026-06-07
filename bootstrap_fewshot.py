"""
Sample candidate few-shot examples: generate a pool of DIVERSE characters ZERO-SHOT
with the base model and dump them to fewshot_candidates.json (+ readable stdout). Then
we hand-pick the best 2 and hardcode them into fewshot.py. Sampling diverse seeds (and a
warm temperature) gives variety to choose from; curation handles the zero-shot misfires
(e.g. a name coming out as "42").

Run:  python3 bootstrap_fewshot.py     # samples + dumps; pick the best afterward
"""
import json
import worldRefactored as wr

SEEDS = [
    ("A neon cyberpunk megacity", "vampire", "male"),
    ("A cozy kingdom of living teaware", "sentient teapot", "nonbinary"),
    ("A bioluminescent deep-sea trench", "anglerfish priest", "female"),
    ("A clockwork city adrift in the clouds", "brass automaton", "female"),
    ("A haunted Victorian manor", "ghost child", "male"),
    ("A desert of singing glass", "sand wraith", "nonbinary"),
    ("A library the size of a continent", "living book", "female"),
    ("A mushroom forest under a red moon", "giant isopod", "female"),
    ("A war-torn floating archipelago", "storm priestess", "female"),
    ("A frozen generation starship", "android botanist", "nonbinary"),
]
SAMPLES_PER_SEED = 1


def main():
    s = wr.LlamaServer(timeout=120, retries=3)  # use the server's configured temperature
    pool = []
    for world, species, gender in SEEDS:
        for _ in range(SAMPLES_PER_SEED):
            try:
                c = wr.Character(s, world, gender, species=species, examples=[])  # zero-shot
                d = {"world": world, "species": species, "gender": gender, "name": c.name,
                     "appearance": c.appearance, "personality": c.personality,
                     "backstory": c.backstory}
                pool.append(d)
                with open("fewshot_candidates.json", "w") as f:           # incremental save
                    json.dump(pool, f, indent=2, ensure_ascii=False)
                print(f"\n[{len(pool)}] {d['name']}  ({species} — {world})")
                print(f"    appearance:  {d['appearance']}")
                print(f"    personality: {d['personality']}")
                print(f"    backstory:   {d['backstory']}", flush=True)
            except Exception as e:
                print(f"err: {world}: {e!r}", flush=True)
    print(f"\nDONE: {len(pool)} candidates -> fewshot_candidates.json. "
          f"Pick the best 2 to bake into fewshot.py.", flush=True)


if __name__ == "__main__":
    main()
