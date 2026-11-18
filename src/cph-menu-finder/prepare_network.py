"""Build the ingredient co-occurrence network from the hand-verified Wolt ingredients, as an
edge list for backboning.py and network.ipynb.

Two ingredients are linked when they appear in the same menu item, and the weight of the link
is the number of items they share. The network is undirected, so each pair is stored once
(src < trg alphabetically), and an ingredient is never linked to itself.

Steps:
    1. Read item_ingredients.csv and ingredients.csv from data/wolt/copenhagen/.
    2. Keep only the raw ingredients marked as food (is_ingredient == 1) and replace each with
       its verified name (ingredient_verified). Several raw spellings can map to the same
       verified ingredient, so duplicates per item are dropped afterwards.
    3. Self-join the (item_id, ingredient) table on item_id to get every ingredient pair per
       item, and count in how many items each pair appears.

Output:
    data/network/edges-<YYYY-MM-DD>.csv    one row per ingredient pair (src, trg, weight),
                                            sorted by weight, most common pair first

The file name carries today's date, so earlier edge lists are kept and a re-run on the same
day overwrites that day's file.

@author: Rakul Tórgarð
@review: Rakul Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    python prepare_network.py
"""

from datetime import date

import pandas as pd
from config import *


def read_data():
    """Return one row per (item_id, ingredient), using verified ingredient names and only the
    raw ingredients that were annotated as food."""
    path = WOLT_DATA_DIR / 'copenhagen'

    item_ingredients = pd.read_csv(path / 'item_ingredients.csv')
    # dropna() leaves out ingredients that have not been reviewed yet (empty cells)
    ingredients_raw = pd.read_csv(path / 'ingredients.csv').dropna()
    ingredients_food = ingredients_raw[ingredients_raw['is_ingredient'] == 1.0]
    dishes_ingredients = (
        item_ingredients
        .merge(ingredients_food[["ingredient_raw", "ingredient_verified"]], on="ingredient_raw")
        .rename(columns={"ingredient_verified": "ingredient"})[["item_id", "ingredient"]]
        .drop_duplicates()     
        .reset_index(drop=True)
    )

    return dishes_ingredients


def write_to_edges(df):
    """Count how many items each pair of ingredients shares and write the weighted, undirected
    edge list (src, trg, weight) to NETWORK_DATA_DIR."""
    # self-join on item_id: every combination of two ingredients within the same item
    pairs = (
        df
        .merge(df, on="item_id", suffixes=("_a", "_b"))
        .rename(columns={"ingredient_a": "src", "ingredient_b": "trg"})
    )
    pairs = pairs[pairs["src"] < pairs["trg"]]

    # weight = number of items the two ingredients appear in together
    edges = (
        pairs.groupby(["src", "trg"])
        .size()
        .reset_index(name="weight")
        .sort_values(by='weight', ascending=False)
    )

    edges.to_csv(NETWORK_DATA_DIR / f"edges-{date.today()}.csv", index=False)

    print(f"Wrote {len(df)} edges to -> {NETWORK_DATA_DIR / f"edges-{date.today()}.csv"}")


def main():
    df = read_data()
    write_to_edges(df)


if __name__ == "__main__":
    main()
