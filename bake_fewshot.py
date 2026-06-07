"""
Bake the 2 curated few-shot characters (Danny + The Book of the Shifting Shadows,
hand-picked from bootstrap_fewshot.py's pool) into fewshot.py, which worldRefactored
imports for character-field priming. (The schedule deliberately uses no few-shot — an
example there just primes the model to stay in one place.)
"""
import json

CHOSEN = [
    {"world": "A haunted Victorian manor", "species": "ghost child", "gender": "male",
     "name": "Danny",
     "appearance": "Danny is a pale child with wispy, untamed hair that seems to flicker like firelight, and haunted, hollow eyes that wander the space where his soul has fled",
     "personality": "A chilling whisper with the heart of a forgotten dreamer",
     "backstory": "Danny once held a life that burned bright like a candle in a draft, yet waned under a storm of neglect, molding him into the sorrow-laden specter he is today"},
    {"world": "A library the size of a continent", "species": "living book", "gender": "female",
     "name": "The Book of the Shifting Shadows",
     "appearance": "a towering mass of shifting, writhing pages, each page inhabited by words that dance and change, creating a swirling vortex of sound and color",
     "personality": "wise, unpredictable, and ever-changing, like the books that it is made of",
     "backstory": "Originally crafted by an ancient civilization, the Book of the Shifting Shadows contains stories about the shaping of the world and the inevitability of change"},
]


def main():
    with open("fewshot.py", "w") as f:
        f.write("# Curated few-shot character examples (hand-picked from bootstrap_fewshot.py's pool).\n")
        f.write("# Only the character fields use few-shot; the schedule deliberately does not.\n\n")
        f.write("FEWSHOT_CHARACTERS = " + json.dumps(CHOSEN, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote fewshot.py with {len(CHOSEN)} characters (no schedule few-shot)")


if __name__ == "__main__":
    main()
