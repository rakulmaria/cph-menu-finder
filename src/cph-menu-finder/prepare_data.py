"""Script used to prepare data for LLM input.

@author: Rakul Tórgarð & Claude Code
@review: Rakul Tórgarð

Subcommands:
    full_data   filter a raw Google Maps places file down to operational, "proper"
                restaurants and write it in LLM input format
    sample      randomly sample a fraction of restaurants from a raw Google Maps
                places file and write it in LLM input format

Usage:
    python prepare_data.py full_data copenhagen-bounds-raw.json

    python prepare_data.py sample \
        data/raw_data/copenhagen-bounds-raw.json \
        --fraction 0.1 --seed 42
"""

import argparse
import json
import random
from datetime import date
from pathlib import Path
import pandas as pd
from config import *
from data_cleaner import *

def prepare_full_data(file):
    """
    Reads raw data and prepares it for LLM as input data.
    - filters by operational restaurants only
    - filters away "unproper" restaurant types (hot dog stands etc.)
    - structures in proper input format

    Writes the result to RAW_DATA_DIR / <file stem without -raw>-<today>.json
    """
    df = pd.read_json(RAW_DATA_DIR / file)
    df = filter_dataframe(df)

    # unpack name
    df["name"] = df["displayName"].apply(lambda d: d["text"] if isinstance(d, dict) else d)

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

    output = RAW_DATA_DIR / f"{Path(file).stem.removesuffix('-raw')}-{date.today()}.json"
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write('\n')

    print(f"Kept {len(df)} restaurants -> {output}")

    return df


def prepare_sample(file, fraction=0.1, seed=42):
    """
    Randomly sample a fraction of restaurants from a raw Google Maps places
    file and reduce them to the fields llm_menu_finder.py expects (id, name,
    formattedAddress, websiteUri).

    Author: @Claude Code
    Reviewed: @Rakul M. H. Tórgarð

    Writes the sample to RAW_DATA_DIR / copenhagen-<pct>pct-sample-input-<today>.json,
    ready to point RESTAURANT_FILE at directly in llm_menu_finder.py
    """
    file = Path(file)
    output = RAW_DATA_DIR / f"copenhagen-{fraction * 100:g}pct-sample-input-{date.today()}.json"

    restaurants = json.loads(file.read_text(encoding="utf-8"))
    n = round(len(restaurants) * fraction)

    rng = random.Random(seed)
    sample = rng.sample(restaurants, n)

    df = pd.DataFrame(sample)

    if "displayName" in df.columns:
        df["name"] = df["displayName"].apply(lambda d: d["text"] if isinstance(d, dict) else d)

    df = df.reindex(columns=["id", "name", "formattedAddress", "websiteUri"])
    df = df.astype(object).where(df.notna(), None)
    records = df.to_dict(orient="records")

    output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Sampled {n}/{len(restaurants)} restaurants ({fraction:.0%}, seed={seed}) -> {output}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    df_parser = subparsers.add_parser("full_data", help="filter raw data to proper, operational restaurants")
    df_parser.add_argument("file", help="raw Google Maps places JSON file name, relative to RAW_DATA_DIR")

    sample_parser = subparsers.add_parser("sample", help="randomly sample a fraction of restaurants")
    sample_parser.add_argument("file", type=Path, help="raw Google Maps places JSON file to sample from")
    sample_parser.add_argument("--fraction", type=float, default=0.1, help="fraction of restaurants to sample (default: %(default)s)")
    sample_parser.add_argument("--seed", type=int, default=42, help="random seed, for a reproducible sample (default: %(default)s)")

    args = parser.parse_args()

    if args.command == "full_data":
        prepare_full_data(args.file)
    elif args.command == "sample":
        prepare_sample(args.input, args.fraction, args.seed)


if __name__ == "__main__":
    main()
