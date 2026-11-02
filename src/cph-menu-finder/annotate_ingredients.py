"""Manual annotation tool for raw_to_ingredient.csv: decide, one raw word at a time, which
ingredient it stands for.

Goes through every raw word that is not reviewed yet, most frequent first (the top 300 words
cover about 70% of all occurrences, so coverage goes up fast at the start). For each word it
shows how many food items it appears in and a few example descriptions, and asks you to:
    keep      the word is an ingredient as it is            (tomat -> tomat)
    map       the word stands for another ingredient        (kyling -> kylling, prawns -> reje)
    drop      the word is not an ingredient                 (fyld, valgfri, stk, x, co2e)

Every decision sets `ingredient` and marks the row `reviewed` = "x". Coverage, the share of
all word occurrences (sum of n) that is reviewed, is shown as you go: stop when it is high
enough (e.g. 85-90%) or when a batch of words barely moves it. Unreviewed words are not
lost, they keep their raw form as `ingredient`.

Progress is saved to disk after every single annotation, so you can quit any time (Ctrl+C or
'q') and re-run the script later to pick up exactly where you left off. 'u' undoes the last
annotation of this session.

raw_to_ingredient.csv is made by raw_to_ingredient.py, which never overwrites what is
annotated here, so the two can be run in any order.

Author: @Claude Code
Reviewed: @Rakul M. H. Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    python annotate_ingredients.py
    python annotate_ingredients.py --area vesterbro
"""

import argparse
import re
import sys

import pandas as pd
from config import WOLT_DATA_DIR
from raw_to_ingredient import COLUMNS, load_food_items

N_EXAMPLES = 3      # example descriptions shown per word
EXAMPLE_WIDTH = 110


def load_table(path) -> pd.DataFrame:
    # keep_default_na=False so dropped words (empty `ingredient`) are read as "" instead of NaN
    table = pd.read_csv(path, keep_default_na=False)
    if "reviewed" not in table:   # csv from before the column existed
        table["reviewed"] = ""
    return table[COLUMNS]


def save_table(table: pd.DataFrame, path) -> None:
    table.to_csv(path, index=False)


def coverage(table: pd.DataFrame) -> float:
    """Share of all word occurrences (sum of n) whose word is reviewed."""
    total = table.n.sum()
    return table.n[table.reviewed == "x"].sum() / total if total else 0.0


def examples(word: str, stored: str, descriptions: pd.Series) -> list[str]:
    """The stored example first, then other descriptions containing the word. Up to 2 extra letters are allowed,
    since the lemma is often the word minus its ending (agurk -> agurker), but not more (ris should not match ristede)."""
    found = descriptions[descriptions.str.contains(r"\b" + re.escape(word) + r"\w{0,2}\b", flags=re.UNICODE)]
    shown = [stored] + [d for d in found.head(N_EXAMPLES * 3) if d != stored]
    return [re.sub(r"\s+", " ", d)[:EXAMPLE_WIDTH] for d in shown[:N_EXAMPLES] if d]


def prompt_decision(word: str) -> str:
    """Returns the ingredient ("" = drop), or one of the commands s, u, q."""
    prompt = (
        f"  [enter] keep as '{word}'   [d] drop (not an ingredient)   or type the ingredient it stands for\n"
        "  [s] skip for now   [u] undo last   [q] save and quit\n"
        "  > "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice == "":
            return word
        if choice == "d":
            return ""
        if choice in ("s", "u", "q"):
            return choice
        if len(choice) == 1:
            print("  Invalid input, try again.\n")
            continue
        return choice


def confirm_new_ingredient(ingredient: str, known: set[str]) -> bool:
    """Catches typos: asks before mapping to an ingredient no other word maps to yet."""
    if ingredient in known:
        return True
    return input(f"  '{ingredient}' is a new ingredient (no other word maps to it yet). Use it? [y/n] > ").strip().lower() == "y"


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate raw_to_ingredient.csv: which ingredient does each raw word stand for?")
    parser.add_argument("--area", default="vesterbro")
    args = parser.parse_args()
    path = WOLT_DATA_DIR / args.area / "raw_to_ingredient.csv"
    if not path.exists():
        sys.exit(f"{path} not found, run raw_to_ingredient.py --area {args.area} first")

    table = load_table(path)
    descriptions = pd.Series(load_food_items(args.area).description.unique())
    remaining = table[(table.reviewed != "x") & (table.n > 0)].sort_values("n", ascending=False).index

    n_done = (table.reviewed == "x").sum()
    print(f"Area: {args.area}. {n_done}/{len(table)} words reviewed, coverage {coverage(table):.1%}. "
          f"{len(remaining)} remaining.\n")

    history = []   # (row index, previous ingredient, previous reviewed), for undo
    i = 0
    while i < len(remaining):
        idx = remaining[i]
        row = table.loc[idx]

        print("\n" + "=" * 70)
        print(f"[{n_done + 1}/{len(table)}]  coverage {coverage(table):.1%}")
        print(f"WORD: {row.ingredient_raw}   (in {row.n} food items)")
        if row.ingredient != row.ingredient_raw:
            print(f"  currently mapped to: {row.ingredient or '(dropped)'}")
        for ex in examples(row.ingredient_raw, row.example, descriptions):
            print(f"    - {ex}")
        print("=" * 70)

        result = prompt_decision(row.ingredient_raw)

        if result == "q":
            print(f"\nSaved. {n_done}/{len(table)} reviewed, coverage {coverage(table):.1%}. Stored at {path}")
            sys.exit(0)
        if result == "s":
            i += 1
            continue
        if result == "u":
            if not history:
                print("  Nothing to undo in this session.")
                continue
            prev_idx, prev_ingredient, prev_reviewed = history.pop()
            table.loc[prev_idx, ["ingredient", "reviewed"]] = [prev_ingredient, prev_reviewed]
            save_table(table, path)
            n_done -= 1
            i = list(remaining).index(prev_idx)   # back to the undone word
            print(f"  Undid '{table.loc[prev_idx, 'ingredient_raw']}'.")
            continue

        known = set(table.ingredient[(table.reviewed == "x") & (table.ingredient != "")])
        if result not in ("", row.ingredient_raw) and not confirm_new_ingredient(result, known):
            continue   # ask again for the same word

        history.append((idx, row.ingredient, row.reviewed))
        table.loc[idx, ["ingredient", "reviewed"]] = [result, "x"]
        save_table(table, path)
        n_done += 1
        i += 1

    print(f"\nAll words reviewed! {n_done}/{len(table)}, coverage {coverage(table):.1%}. Saved to {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped. Progress up to the last annotation is stored in raw_to_ingredient.csv")
