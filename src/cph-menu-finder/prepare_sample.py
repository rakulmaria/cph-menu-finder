"""Randomly sample a fraction of restaurants from a raw Google Maps places
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

import argparse
import json
import random
from pathlib import Path

from llm_gateway_client import parse_restaurants


def main():
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

    parsed = parse_restaurants(raw_path)
    args.output.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Sampled {n}/{len(restaurants)} restaurants ({args.fraction:.0%}, seed={args.seed})")
    print(f"  raw sample -> {raw_path}")
    print(f"  parsed     -> {args.output}")


if __name__ == "__main__":
    main()
