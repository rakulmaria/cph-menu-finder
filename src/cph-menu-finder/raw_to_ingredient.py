"""Build the raw_to_ingredient.csv lookup table from the Wolt menu descriptions.

Steps, for one area:
    1. items      read items.csv, normalise the text and keep food items only
                  (drinks, dips and engangsservice are dropped by category)
    2. lemmas     lemmatise each unique description with spaCy (da_core_news_trf),
                  keeping nouns, proper nouns and X (the Danish model tags many foreign
                  food words as X), minus Danish and English stop words
    3. table      explode to one row per (item, lemma) and count how many food items
                  each lemma appears in

The table is meant to be annotated by hand (see annotate_ingredients.py): change `ingredient`
where a raw word should map to something else (kyling -> kylling, prawns -> reje), or leave it
empty to drop a non-food word (valgfri, stk), and mark the row `reviewed` = "x". Re-running never
overwrites those edits: existing rows keep their `ingredient` and `reviewed`, only `n` and
`example` are refreshed, and new raw words are added unreviewed with `ingredient` = `ingredient_raw`.
Raw words that no longer occur keep their row with n = 0.

Output (in data/wolt/<area>/):
    raw_to_ingredient.csv   ingredient_raw, n, ingredient, reviewed, example

@author: Rakul Tórgarð & Claude Code
@review: Rakul Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    python raw_to_ingredient.py
    python raw_to_ingredient.py --area vesterbro
    python raw_to_ingredient.py --area copenhagen
"""

import argparse
import re

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from config import WOLT_DATA_DIR

# a category is a drink if one of its words IS one of these ...
DRINK_WORDS = {"øl", "kaffe", "te", "lemonade", "lassi", "frapino", "drink", "drinks", "jarritos"}
# ... or ENDS WITH one of these (Danish compounds: sodavand, rødvin, mælkedrikke, energidrik, ...)
DRINK_ENDINGS = ("drik", "drikke", "drikkevare", "drikkevarer", "vand", "vin", "vine",
                 "juice", "juicer", "smoothie", "smoothies", "shake", "shakes")
# ... unless it also has one of these words (smoothie bowls, "hovedret + valgfri drik")
NOT_DRINK_WORDS = {"bowl", "bowls", "hovedret", "hovedretter", "menu", "menuer"}
# whole category names that are not food ("burger med fries & dip" stays food)
OTHER = {"dip", "dips", "crust dip", "crust dips", "dyppelse", "engangsservice"}

# words spaCy splits apart that belong together
MULTI_WORD = {"pommes|frites": "pommes frites"}

COLUMNS = ["ingredient_raw", "n", "ingredient", "reviewed", "example"]


def normalise(s: str) -> str:
    return s.lower().strip().rstrip(".")


def is_drink(category: str) -> bool:
    words = re.findall(r"\w+", category)
    if any(w in NOT_DRINK_WORDS for w in words):
        return False
    return any(w in DRINK_WORDS or w.endswith(DRINK_ENDINGS) for w in words)


def is_other(category: str) -> bool:
    return category in OTHER or category.startswith("engangsservice")


def load_food_items(area: str) -> pd.DataFrame:
    items = pd.read_csv(WOLT_DATA_DIR / area / "items.csv")
    for col in ["category", "item", "description"]:
        items[col] = items[col].fillna("").map(normalise)
    is_food = ~(items.category.map(is_drink) | items.category.map(is_other))
    return items[is_food & (items.description != "")]


def lemmatise_doc(descriptions: list[str]) -> dict[str, list[str]]:
    """description -> list of lemmas, joining the words in MULTI_WORD back together."""
    nlp = spacy.load("da_core_news_trf")
    lemmas = {}
    for desc, doc in zip(descriptions, nlp.pipe(descriptions, batch_size=64)):
        words = [t.lemma_.lower() for t in doc
            if not (t.is_stop or t.is_punct or t.is_space)
            and t.pos_ in {"NOUN", "PROPN", "X"}
            and t.lemma_.lower() not in EN_STOP]
        joined = "|".join(words)
        for split, whole in MULTI_WORD.items():
            joined = joined.replace(split, whole)
        lemmas[desc] = joined.split("|") if joined else []
    return lemmas


def count_raw(items: pd.DataFrame, lemmas: dict[str, list[str]]) -> pd.DataFrame:
    """One row per raw word: in how many food items it occurs, plus one example description."""
    exploded = (
        items[["description"]]
        .reset_index(names="row")    # one id per food item, so repeated descriptions (chains) count per item
        .assign(ingredient_raw=lambda d: d.description.map(lemmas))
        .explode("ingredient_raw")
        .dropna(subset=["ingredient_raw"])
        .drop_duplicates(["row", "ingredient_raw"])   # same word twice in one item counts once
    )
    table = (
        exploded.groupby("ingredient_raw")
        .agg(n=("row", "size"), example=("description", "first"))
        .reset_index()
    )
    # one line, max 120 characters, so the csv stays easy to edit by hand
    table["example"] = table.example.str.replace(r"\s+", " ", regex=True, flags=re.UNICODE).str.slice(0, 120)
    return table


def merge_with_existing(fresh: pd.DataFrame, path) -> pd.DataFrame:
    """Keeps every hand-edited `ingredient` and `reviewed`; new raw words map to themselves, unreviewed."""
    if not path.exists():
        return fresh.assign(ingredient=fresh.ingredient_raw, reviewed="")[COLUMNS]
    existing = pd.read_csv(path, keep_default_na=False)   # empty `ingredient` stays "", not NaN
    if "reviewed" not in existing:                         # csv from before the column existed
        existing["reviewed"] = ""
    merged = existing[["ingredient_raw", "ingredient", "reviewed"]].merge(fresh, on="ingredient_raw", how="outer")
    merged["ingredient"] = merged.ingredient.fillna(merged.ingredient_raw)
    merged["reviewed"] = merged.reviewed.fillna("")
    merged["n"] = merged.n.fillna(0).astype(int)
    merged["example"] = merged.example.fillna("")
    return merged[COLUMNS]


def main():
    """Lemmatises the food descriptions of one area and writes/updates raw_to_ingredient.csv."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="vesterbro")
    args = parser.parse_args()
    path = WOLT_DATA_DIR / args.area / "raw_to_ingredient.csv"

    items = load_food_items(args.area)
    descriptions = list(items.description.unique())
    print(f"{len(items)} food items with a description, {len(descriptions)} unique descriptions")

    lemmas = lemmatise_doc(descriptions)
    table = merge_with_existing(count_raw(items, lemmas), path)
    table = table.sort_values(["n", "ingredient_raw"], ascending=[False, True])
    table.to_csv(path, index=False)

    edited = (table.ingredient != table.ingredient_raw).sum()
    print(f"{len(table)} raw words ({edited} hand-edited) -> {path}")


if __name__ == "__main__":
    main()
