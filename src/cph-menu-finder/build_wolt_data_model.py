"""Build the ingredient tables from the Wolt menu descriptions.

Data model, per area (all in data/wolt/<area>/):

    venues.csv              wolt_id, name, ...                          one row per venue
        | 1 -> N            (items.wolt_id)
    items.csv               item_id, wolt_id, ..., description          one row per menu item
        | N -> M            (resolved by item_ingredients)
    item_ingredients.csv    item_id, description_raw, ingredient_raw    one row per (item, raw ingredient)
        | N -> 1            (ingredient_raw)
    ingredients.csv         ingredient_raw, n, ingredient_verified,     one row per raw ingredient
                            is_ingredient

Steps:
    1. items      read items.csv, keep food items with a description (drinks, dips and
                  engangsservice are dropped by category), lowercase the description
    2. lemmas     lemmatise each unique description with spaCy (da_core_news_trf): drop stop,
                  punctuation and space tokens, keep NOUN, PROPN and X (the Danish model tags
                  many English/foreign food words as X), minus English stop words, then join
                  words that are one ingredient when they stand directly next to each other:
                  the pairs in MULTI_WORD (pommes + frites -> pommes frites) and any word
                  before a HEAD_WORD (teriyaki + sauce -> teriyaki sauce)
    3. explode    item_ingredients.csv: one row per item and raw ingredient (each raw ingredient
                  at most once per item). Rebuilt from scratch on every run.
    4. lookup     ingredients.csv: every raw ingredient once, with n = the number of food items
                  it appears in, most frequent first. New ones are added unreviewed (empty
                  ingredient_verified and is_ingredient); existing annotations are never changed,
                  so they survive re-runs (only n is recounted). Raw ingredients that no longer
                  occur keep their row with n = 0.

ingredient_verified and is_ingredient are filled in by hand with annotate_ingredients.py:
    ingredient_verified   the raw word as it is (tomat -> tomat) or spell-corrected / mapped
                          (tomate -> tomat, kyling -> kylling)
    is_ingredient         1 = an ingredient, 0 = not one (kilo, co2e, stk), "" = not reviewed yet
Many raw ingredients can share one verified ingredient (kyling, kyllingebryst -> kylling).

@author: Rakul Tórgarð & Claude Code
@review: Rakul Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    python build_wolt_data_model.py
    python build_wolt_data_model.py --area copenhagen
"""

import argparse
import re

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP
from tqdm import tqdm
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

# consecutive lemmas spaCy splits apart that are one ingredient
MULTI_WORD = {
    ("pommes", "frites"): "pommes frites",
    ("creme", "fraiche"): "creme fraiche",
    ("grana", "padano"): "grana padano",
}
# catch-all: "<word> sauce" -> one ingredient, for any kept word directly before it (teriyaki sauce, goma sauce)
HEAD_WORDS = {"sauce"}

ITEM_INGREDIENT_COLUMNS = ["item_id", "description_raw", "ingredient_raw"]
INGREDIENT_COLUMNS = ["ingredient_raw", "n", "ingredient_verified", "is_ingredient"]
ANNOTATED = ["ingredient_verified", "is_ingredient"]   # the hand-made columns, never overwritten


def read_ingredients(path) -> pd.DataFrame:
    """ingredients.csv with unreviewed cells as "" (not NaN) and is_ingredient as the text "1"/"0"/""."""
    table = pd.read_csv(path, keep_default_na=False, dtype={"is_ingredient": str})
    for col in ANNOTATED:
        if col not in table:   # csv from before the column existed
            table[col] = ""
    return table


def paths(area: str) -> tuple:
    """(item_ingredients.csv, ingredients.csv) of an area."""
    return WOLT_DATA_DIR / area / "item_ingredients.csv", WOLT_DATA_DIR / area / "ingredients.csv"


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
    """Food items with a description. `description_raw` is as on Wolt, `description` is normalised."""
    items = pd.read_csv(WOLT_DATA_DIR / area / "items.csv")
    items["description_raw"] = items.description.fillna("")
    for col in ["category", "item", "description"]:
        items[col] = items[col].fillna("").map(normalise)
    is_food = ~(items.category.map(is_drink) | items.category.map(is_other))
    return items[is_food & (items.description != "")]


def join_multi_word(words: list[tuple[int, str]]) -> list[str]:
    """Joins words that are one ingredient, given as (token position, lemma). Only words directly
    next to each other in the text are joined, so "ris og sauce" stays [ris, sauce]:
        a MULTI_WORD pair       [pommes, frites]   -> [pommes frites]
        any word + a HEAD_WORD  [teriyaki, sauce]  -> [teriyaki sauce]"""
    joined, last = [], None   # last = token position of the last word in `joined`
    for i, w in words:
        prev = joined[-1] if joined and last == i - 1 else None
        if prev is not None and (prev, w) in MULTI_WORD:
            joined[-1] = MULTI_WORD[(prev, w)]
        elif prev is not None and w in HEAD_WORDS and not prev.endswith(w):   # not "sauce sauce"
            joined[-1] = f"{prev} {w}"
        else:
            joined.append(w)
        last = i
    return joined


def lemmatise_doc(descriptions: list[str]) -> dict[str, list[str]]:
    """description -> its raw ingredients (lemmas), in order."""
    on_gpu = spacy.prefer_gpu()   # must come before spacy.load; falls back to CPU if there is no GPU
    print(f"spaCy on {'GPU' if on_gpu else 'CPU'}")
    nlp = spacy.load("da_core_news_trf")
    lemmas = {}
    docs = nlp.pipe(descriptions, batch_size=64)
    for desc, doc in tqdm(zip(descriptions, docs), total=len(descriptions), desc="lemmatising", unit="desc"):
        words = [(t.i, t.lemma_.lower()) for t in doc
                 if not (t.is_stop or t.is_punct or t.is_space)
                 and t.pos_ in {"NOUN", "PROPN", "X"}
                 and t.lemma_.lower() not in EN_STOP]
        lemmas[desc] = join_multi_word(words)
    return lemmas


def explode(items: pd.DataFrame, lemmas: dict[str, list[str]]) -> pd.DataFrame:
    """item_ingredients: one row per (item, raw ingredient); items without any raw ingredient drop out."""
    return (
        items.assign(ingredient_raw=items.description.map(lemmas))
        .explode("ingredient_raw")
        .dropna(subset=["ingredient_raw"])
        .drop_duplicates(["item_id", "ingredient_raw"])   # same word twice in one item counts once
        [ITEM_INGREDIENT_COLUMNS]
        .reset_index(drop=True)
    )


def update_ingredients(item_ingredients: pd.DataFrame, path) -> pd.DataFrame:
    """ingredients: existing annotations kept, new raw ingredients added unverified, n recounted,
    most frequent first."""
    counts = item_ingredients.ingredient_raw.value_counts().rename("n").reset_index()
    if path.exists():
        existing = read_ingredients(path)
        merged = existing[["ingredient_raw", *ANNOTATED]].merge(counts, on="ingredient_raw", how="outer")
    else:
        merged = counts.assign(ingredient_verified="", is_ingredient="")
    merged[ANNOTATED] = merged[ANNOTATED].fillna("")
    merged["n"] = merged.n.fillna(0).astype(int)   # raw ingredients that no longer occur keep their row with n = 0
    return merged.sort_values(["n", "ingredient_raw"], ascending=[False, True])[INGREDIENT_COLUMNS]


def main():
    """Lemmatises the food descriptions of one area and writes item_ingredients.csv and ingredients.csv."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", default="vesterbro")
    args = parser.parse_args()
    item_ingredients_path, ingredients_path = paths(args.area)

    items = load_food_items(args.area)
    descriptions = list(items.description.unique())
    print(f"{len(items)} food items with a description, {len(descriptions)} unique descriptions")

    item_ingredients = explode(items, lemmatise_doc(descriptions))
    item_ingredients.to_csv(item_ingredients_path, index=False)
    print(f"{len(item_ingredients)} (item, raw ingredient) rows from {item_ingredients.item_id.nunique()} items "
          f"-> {item_ingredients_path}")

    ingredients = update_ingredients(item_ingredients, ingredients_path)
    ingredients.to_csv(ingredients_path, index=False)
    reviewed = (ingredients.is_ingredient != "").sum()
    print(f"{len(ingredients)} raw ingredients ({reviewed} reviewed) -> {ingredients_path}")


if __name__ == "__main__":
    main()
