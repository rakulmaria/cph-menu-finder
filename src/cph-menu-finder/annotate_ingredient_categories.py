"""Manual annotation tool for categories.csv: go through the verified ingredients one at
a time and give each one a category, following the Annotation Codebook for ingredient categories
(cph-menu-finder.wiki/Annotation-Codebook-ingredients-categories.md).

There are 18 categories: the 14 of Ahn et al. (2011), "Flavor network and the principles of food
pairing", plus four of our own (15-18). The categories are stored in Danish, like the
ingredients; the codebook gives the paper's English name next to each.
    [1]  frugt                      [10] grøntsager
    [2]  mejeriprodukter            [11] blomster
    [3]  krydderier                 [12] animalske produkter
    [4]  alkoholiske drikkevarer    [13] planter
    [5]  nødder og frø              [14] korn
    [6]  fisk og skaldyr            [15] sammensatte
    [7]  kød                        [16] brød, pasta og dej
    [8]  krydderurter               [17] bælgfrugter
    [9]  planteprodukter            [18] ikke-alkoholiske drikkevarer

Each ingredient gets exactly one category, chosen with a number key:
    tomat        -> [10] grøntsager
    mozzarella   -> [2]  mejeriprodukter
    pesto        -> [15] sammensatte
    burgerbolle  -> [16] brød, pasta og dej
    kikærter     -> [17] bælgfrugter
    cola         -> [18] ikke-alkoholiske drikkevarer

The decision is written to the column `category` of categories.csv. The column is
added on the first run; an empty cell means not reviewed yet. Ingredients are shown in file
order, and the script continues with the next unreviewed one when it is run again.

Progress is saved to disk after every single annotation, so you can quit any time (Ctrl+C or
'q') and re-run the script later to pick up exactly where you left off. 's' skips an ingredient
for now (it is shown again next run), 'u' undoes the last annotation of this session.

Author: @Claude Code
Reviewed: @Rakul M. H. Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    python annotate_ingredient_categories.py
    python annotate_ingredient_categories.py --area vesterbro
"""

import argparse
import sys

import pandas as pd
from config import WOLT_DATA_DIR

FILE_NAME = "categories.csv"

# The order of the codebook: the 14 categories of the paper first, then our own four. Danish, like the ingredients
CATEGORIES = {
    "1": "frugt",
    "2": "mejeriprodukter",
    "3": "krydderier",
    "4": "alkoholiske drikkevarer",
    "5": "nødder og frø",
    "6": "fisk og skaldyr",
    "7": "kød",
    "8": "krydderurter",
    "9": "planteprodukter",
    "10": "grøntsager",
    "11": "blomster",
    "12": "animalske produkter",
    "13": "planter",
    "14": "korn",
    "15": "sammensatte",
    "16": "brød, pasta og dej",
    "17": "bælgfrugter",
    "18": "ikke-alkoholiske drikkevarer"
}
MENU_COLUMNS = 4


def load_ingredients(path) -> pd.DataFrame:
    # keep_default_na=False so unreviewed categories are read as "" instead of NaN
    ingredients = pd.read_csv(path, keep_default_na=False)
    if "category" not in ingredients.columns:
        ingredients["category"] = ""   # first run: add the column, every ingredient unreviewed
    return ingredients


def save_ingredients(ingredients: pd.DataFrame, path) -> None:
    ingredients.to_csv(path, index=False)


def is_reviewed(ingredients: pd.DataFrame) -> pd.Series:
    return ingredients.category != ""


def category_menu() -> str:
    """The categories as a numbered grid, MENU_COLUMNS per line."""
    cells = [f"[{key:>2}] {name:<26}" for key, name in CATEGORIES.items()]
    rows = [cells[i:i + MENU_COLUMNS] for i in range(0, len(cells), MENU_COLUMNS)]
    return "\n".join("  " + "".join(row).rstrip() for row in rows)


def prompt_category() -> str:
    """Returns a category, or one of the commands s, u, q."""
    prompt = (
        f"{category_menu()}\n"
        "  [s] skip for now   [u] undo last   [q] save and quit\n"
        "  > "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice in CATEGORIES:
            return CATEGORIES[choice]
        if choice in ("s", "u", "q"):
            return choice
        print("  Invalid input, try again.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate categories.csv: which category is each ingredient?")
    parser.add_argument("--area", default="copenhagen")
    args = parser.parse_args()
    path = WOLT_DATA_DIR / args.area / FILE_NAME
    if not path.exists():
        sys.exit(f"{path} not found, export the verified ingredients for {args.area} first (network.ipynb)")

    ingredients = load_ingredients(path)
    save_ingredients(ingredients, path)   # writes the new category column on the first run
    remaining = ingredients[~is_reviewed(ingredients)].index

    n_done = is_reviewed(ingredients).sum()
    print(f"Area: {args.area}. {n_done}/{len(ingredients)} ingredients categorised. {len(remaining)} remaining.\n")

    history = []   # (row index, previous category), for undo
    i = 0
    while i < len(remaining):
        idx = remaining[i]

        print("\n" + "=" * 70)
        print(f"[{n_done + 1}/{len(ingredients)}]")
        print(f"INGREDIENT: {ingredients.loc[idx, 'ingredient_verified']}")
        print("=" * 70)

        result = prompt_category()

        if result == "q":
            print(f"\nSaved. {n_done}/{len(ingredients)} categorised. Stored at {path}")
            sys.exit(0)
        if result == "s":
            i += 1
            continue
        if result == "u":
            if not history:
                print("  Nothing to undo in this session.")
                continue
            prev_idx, prev_category = history.pop()
            ingredients.loc[prev_idx, "category"] = prev_category
            save_ingredients(ingredients, path)
            n_done -= 1
            i = list(remaining).index(prev_idx)   # back to the undone ingredient
            print(f"  Undid '{ingredients.loc[prev_idx, 'ingredient_verified']}'.")
            continue

        history.append((idx, ingredients.loc[idx, "category"]))
        ingredients.loc[idx, "category"] = result
        save_ingredients(ingredients, path)
        print(f"  -> {result}")
        n_done += 1
        i += 1

    print(f"\nAll ingredients reviewed this run! {n_done}/{len(ingredients)} categorised. Saved to {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nStopped. Progress up to the last annotation is stored in {FILE_NAME}")
