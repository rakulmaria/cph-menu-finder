"""Manual annotation tool for ingredients.csv: go through the raw ingredients one at a time and
fill in `ingredient_verified` and `is_ingredient`.

Goes through every raw ingredient that is not reviewed yet (empty `is_ingredient`), most
frequent first (the most frequent few hundred words cover most occurrences, so coverage goes
up fast at the start). For each word it shows how many food items it appears in and a few of
their descriptions (from item_ingredients.csv), and asks you to:
    keep      [enter]   the word is an ingredient as it is      tomat  -> tomat,   is_ingredient 1
    correct   type it   the word is an ingredient, spelled /    tomate -> tomat,   is_ingredient 1
                        named differently                       prawns -> reje
    drop      [d]       the word is not an ingredient           kilo   -> kilo,    is_ingredient 0
                                                                (co2e, stk, sauce, valgfri)

Coverage, the share of all (item, raw ingredient) rows whose raw ingredient is reviewed, is
shown as you go: stop when it is high enough (e.g. 85-90%) or when a batch of words barely
moves it.

Progress is saved to disk after every single annotation, so you can quit any time (Ctrl+C or
'q') and re-run the script later to pick up exactly where you left off. 'u' undoes the last
annotation of this session.

ingredients.csv and item_ingredients.csv are made by build_wolt_data_model.py, which never
overwrites what is annotated here, so the two can be run in any order.

Author: @Claude Code
Reviewed: @Rakul M. H. Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    python annotate_ingredients.py
    python annotate_ingredients.py --area copenhagen
"""

import argparse
import re
import sys

import pandas as pd
from build_wolt_data_model import ANNOTATED, INGREDIENT_COLUMNS, paths, read_ingredients

N_EXAMPLES = 3      # example descriptions shown per word
EXAMPLE_WIDTH = 110


def save_ingredients(ingredients: pd.DataFrame, path) -> None:
    ingredients.to_csv(path, index=False)


def is_reviewed(ingredients: pd.DataFrame) -> pd.Series:
    return ingredients.is_ingredient != ""


def coverage(ingredients: pd.DataFrame, n: pd.Series) -> float:
    """Share of all (item, raw ingredient) rows whose raw ingredient is reviewed."""
    reviewed = ingredients.ingredient_raw[is_reviewed(ingredients)]
    total = n.sum()
    return n[reviewed].sum() / total if total else 0.0


def examples(descriptions: list[str]) -> list[str]:
    """A few distinct descriptions on one line each."""
    shown = list(dict.fromkeys(re.sub(r"\s+", " ", d).strip() for d in descriptions))
    return [d[:EXAMPLE_WIDTH] for d in shown[:N_EXAMPLES]]


def prompt_decision(word: str) -> tuple[str, str] | str:
    """Returns (ingredient_verified, is_ingredient), or one of the commands s, u, q."""
    prompt = (
        f"  [enter] keep as '{word}'   [d] drop (not an ingredient)   or type the correct spelling / ingredient\n"
        "  [s] skip for now   [u] undo last   [q] save and quit\n"
        "  > "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice == "":
            return word, "1"
        if choice == "d":
            return word, "0"
        if choice in ("s", "u", "q"):
            return choice
        if len(choice) == 1:
            print("  Invalid input, try again.\n")
            continue
        return choice, "1"


def confirm_new_ingredient(ingredient: str, known: set[str]) -> bool:
    """Catches typos: asks before correcting to an ingredient no other word maps to yet."""
    if ingredient in known:
        return True
    return input(f"  '{ingredient}' is a new ingredient (no other word maps to it yet). Use it? [y/n] > ").strip().lower() == "y"


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate ingredients.csv: is each raw word an ingredient, and which one?")
    parser.add_argument("--area", default="vesterbro")
    args = parser.parse_args()
    item_ingredients_path, path = paths(args.area)
    if not path.exists() or not item_ingredients_path.exists():
        sys.exit(f"{path} not found, run build_wolt_data_model.py --area {args.area} first")

    ingredients = read_ingredients(path)[INGREDIENT_COLUMNS]
    item_ingredients = pd.read_csv(item_ingredients_path, keep_default_na=False)
    n = ingredients.set_index("ingredient_raw").n                          # food items per raw ingredient
    descriptions = item_ingredients.groupby("ingredient_raw").description_raw.apply(list)

    remaining = ingredients[~is_reviewed(ingredients) & (ingredients.n > 0)] \
        .sort_values("n", ascending=False).index

    n_done = is_reviewed(ingredients).sum()
    print(f"Area: {args.area}. {n_done}/{len(ingredients)} words reviewed, coverage {coverage(ingredients, n):.1%}. "
          f"{len(remaining)} remaining.\n")

    history = []   # (row index, previous [ingredient_verified, is_ingredient]), for undo
    i = 0
    while i < len(remaining):
        idx = remaining[i]
        word = ingredients.loc[idx, "ingredient_raw"]

        print("\n" + "=" * 70)
        print(f"[{n_done + 1}/{len(ingredients)}]  coverage {coverage(ingredients, n):.1%}")
        print(f"WORD: {word}   (in {n[word]} food items)")
        for ex in examples(descriptions[word]):
            print(f"    - {ex}")
        print("=" * 70)

        result = prompt_decision(word)

        if result == "q":
            print(f"\nSaved. {n_done}/{len(ingredients)} reviewed, coverage {coverage(ingredients, n):.1%}. Stored at {path}")
            sys.exit(0)
        if result == "s":
            i += 1
            continue
        if result == "u":
            if not history:
                print("  Nothing to undo in this session.")
                continue
            prev_idx, prev_values = history.pop()
            ingredients.loc[prev_idx, ANNOTATED] = prev_values
            save_ingredients(ingredients, path)
            n_done -= 1
            i = list(remaining).index(prev_idx)   # back to the undone word
            print(f"  Undid '{ingredients.loc[prev_idx, 'ingredient_raw']}'.")
            continue

        verified, is_ingredient = result
        known = set(ingredients.ingredient_verified[ingredients.is_ingredient == "1"])
        if verified != word and not confirm_new_ingredient(verified, known):
            continue   # ask again for the same word

        history.append((idx, list(ingredients.loc[idx, ANNOTATED])))
        ingredients.loc[idx, ANNOTATED] = [verified, is_ingredient]
        save_ingredients(ingredients, path)
        n_done += 1
        i += 1

    print(f"\nAll words reviewed! {n_done}/{len(ingredients)}, coverage {coverage(ingredients, n):.1%}. Saved to {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped. Progress up to the last annotation is stored in ingredients.csv")
