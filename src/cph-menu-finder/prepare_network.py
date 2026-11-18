"""Build the ingredient co-occurrence network from the hand-verified Wolt ingredients, as an
edge list for backboning.py and network.ipynb.

Two ingredients are linked when they appear in the same menu item. The network is undirected,
so each pair is stored once (src < trg alphabetically), and an ingredient is never linked to
itself. The weight of the link depends on --count:
    item    (default) the number of items the two ingredients share.
    venue   the number of venues with at least one item that has both ingredients. Chains and
            copy-pasted menus repeat the same item at many venues (and many times at one
            venue), which inflates item counts and makes the noise-corrected backbone keep
            almost every pair. Counting once per venue removes most of these copies.

Steps:
    1. Read item_ingredients.csv, ingredients.csv and items.csv from data/wolt/copenhagen/.
    2. Keep only the raw ingredients marked as food (is_ingredient == 1) and replace each with
       its verified name (ingredient_verified). Several raw spellings can map to the same
       verified ingredient, so duplicates per item are dropped afterwards.
    3. Self-join the (item_id, ingredient) table on item_id to get every ingredient pair per
       item, and count in how many items (or venues) each pair appears.

Output:
    data/network/edges-<YYYY-MM-DD>.csv          --count item
    data/network/edges-<YYYY-MM-DD>-venue.csv    --count venue
    One row per ingredient pair (src, trg, weight), sorted by weight, most common pair first.

The file name carries today's date, so earlier edge lists are kept and a re-run on the same
day overwrites that day's file.

@author: Rakul Tórgarð
@review: Rakul Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    python prepare_network.py
    python prepare_network.py --count venue
"""

import argparse
import sys
from datetime import date

import pandas as pd
from config import *


def read_data():
    """Return one row per (item_id, ingredient), using verified ingredient names and only the
    raw ingredients that were annotated as food."""
    path = WOLT_DATA_DIR / 'copenhagen'
    for name in ['item_ingredients.csv', 'ingredients.csv', 'items.csv']:
        if not (path / name).exists():
            sys.exit(f"{path / name} not found, run wolt_menus.py and raw_to_ingredient.py first")

    item_ingredients = pd.read_csv(path / 'item_ingredients.csv')
    # dropna() leaves out ingredients that have not been reviewed yet (empty cells)
    ingredients_raw = pd.read_csv(path / 'ingredients.csv').dropna()
    ingredients_food = ingredients_raw[ingredients_raw['is_ingredient'] == 1.0]
    items = pd.read_csv(path / 'items.csv', usecols=["item_id", "wolt_id"])
    dishes_ingredients = (
        item_ingredients
        .merge(ingredients_food[["ingredient_raw", "ingredient_verified"]], on="ingredient_raw")
        .rename(columns={"ingredient_verified": "ingredient"})[["item_id", "ingredient"]]
        .drop_duplicates()
        .merge(items, on="item_id")     # wolt_id = the venue, needed for --count venue
        .reset_index(drop=True)
    )

    return dishes_ingredients


def write_to_edges(df, count="item"):
    """Count how many items (or venues) each pair of ingredients shares and write the weighted,
    undirected edge list (src, trg, weight) to NETWORK_DATA_DIR."""
    # self-join on item_id: every combination of two ingredients within the same item
    pairs = (
        df
        .merge(df[["item_id", "ingredient"]], on="item_id", suffixes=("_a", "_b"))
        .rename(columns={"ingredient_a": "src", "ingredient_b": "trg"})
    )
    pairs = pairs[pairs["src"] < pairs["trg"]]

    # a venue counts once per pair, however many of its items have both ingredients
    if count == "venue":
        pairs = pairs.drop_duplicates(["wolt_id", "src", "trg"])

    # weight = number of items (or venues) the two ingredients appear in together
    edges = (
        pairs.groupby(["src", "trg"])
        .size()
        .reset_index(name="weight")
        .sort_values(by='weight', ascending=False)
    )

    suffix = "" if count == "item" else f"-{count}"
    out = NETWORK_DATA_DIR / f"edges-{date.today()}{suffix}.csv"
    edges.to_csv(out, index=False)

    print(f"Wrote {len(edges)} edges to -> {out}")


def main():
    parser = argparse.ArgumentParser(description="Build the ingredient co-occurrence edge list.")
    parser.add_argument("--count", choices=["item", "venue"], default="item",
                        help="count a pair once per item (default) or once per venue")
    args = parser.parse_args()

    df = read_data()
    write_to_edges(df, args.count)


if __name__ == "__main__":
    main()
