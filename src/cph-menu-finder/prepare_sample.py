"""Script used to prepare data for LLM input
"""

import argparse
import json
import random
from pathlib import Path
import pandas as pd
from config import *

from llm_gateway_client import parse_restaurants

def prepare_df(file):
    """
    Reads raw data and prepares it for LLM as input data.
    - filters by operational restaurants only
    - filters away "unproper" restaurant types (hot dog stands etc.)
    - structures in proper input format
    """
    df = pd.read_json(RAW_DATA_DIR / file)

    # unpack name
    df["name"] = df["displayName"].apply(lambda d: d["text"] if isinstance(d, dict) else d)

    # remove out of order restaurants
    df = df[df["businessStatus"] == "OPERATIONAL"]

    # possible "proper" restaurant types that aren't named restaurant or *_restaurant
    other_types_to_keep = ["bagel_shop", "bistro", "diner", "pizza_delivery"]
    # not considered "proper" restaurant
    discard_restaurants = ["shawarma_restaurant", "hot_dog_restaurant"]

    is_restaurant = df["primaryType"].str.fullmatch(r"(.+_)?restaurant", na=False) | df["primaryType"].isin(other_types_to_keep)
    df = df[is_restaurant & ~df["primaryType"].isin(discard_restaurants)]
    
    cols_to_keep = [
        "id", 
        "name",
        "formattedAddress", 
        "websiteUri",
    ]
    
    # reindex and filter columns
    df = df[cols_to_keep]
    df.reset_index(drop=True, inplace=True)

    records = json.loads(df.to_json(orient='records'))

    with open(RAW_DATA_DIR / f'{file.rstrip('.json')}-{str(date.today())}.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write('\n')

    return df


def prepare_sample():
    """
    Randomly sample a fraction of restaurants from a raw Google Maps places
    file and reduce them to the fields llm_menu_finder.py expects (id, name,
    formattedAddress, websiteUri) -- reusing its own parse_restaurants().

    Author: @Claude Code
    Reviewed: @Rakul M. H. Tórgarð

    Writes two files:
    <output stem>.raw.json  -- the sampled subset, full raw Google Maps schema
                                (kept so the exact sample can be re-derived or
                                inspected later)
    <output>                -- the same restaurants reduced to the schema
                                parse_restaurants() produces, ready to point
                                RESTAURANT_FILE at directly in llm_menu_finder.py

    Usage:
        python prepare_sample.py \
            data/raw_data/copenhagen-bounds-raw.json \
            data/raw_data/copenhagen-10pct-sample-input.json \
            --fraction 0.1 --seed 42
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", type=Path, help="raw Google Maps places JSON file to sample from")
    parser.add_argument("output", type=Path, help="path to write the parsed sample to")
    parser.add_argument("--fraction", type=float, default=0.1, help="fraction of restaurants to sample (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="random seed, for a reproducible sample (default: 42)")
    args = parser.parse_args()

    restaurants = json.loads(args.input.read_text(encoding="utf-8"))
    n = round(len(restaurants) * args.fraction)

    rng = random.Random(args.seed)
    sample = rng.sample(restaurants, n)

    raw_path = args.output.with_name(args.output.stem + "-raw" + args.output.suffix)
    raw_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.read_json(raw_path)

    if "displayName" in df.columns:
        df["name"] = df["displayName"].apply(lambda d: d["text"] if isinstance(d, dict) else d)

    df = df.reindex(columns=["id", "name", "formattedAddress", "websiteUri"])
    df = df.astype(object).where(df.notna(), None)   
    df = df.to_dict(orient="records")

    args.output.write_text(json.dumps(df, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Sampled {n}/{len(restaurants)} restaurants ({args.fraction:.0%}, seed={args.seed})")
    print(f"  raw sample -> {raw_path}")
    print(f"  parsed     -> {args.output}")


def main():
    return


if __name__ == "__main__":
    main()
