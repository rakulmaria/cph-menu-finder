"""Manual annotation tool for menu link quality, following the Annotation Codebook
(cph-menu-finder.wiki/Annotation-Codebook.md).

Goes through every (restaurant, model, response) entry in
full_dataset_disagreement_links_to_annotate.csv one at a time and asks you to
label the response. The model is never shown, and the order of the two models
is shuffled per restaurant (with a fixed seed, so the order is the same every
time the script is restarted). Empty responses are shown as <no-url-found> and
labelled with the codebook's Case 2 labels.

Two modes:
    full   (default)  every restaurant in the file
    sample (--sample) a fixed 10% of the restaurants, stratified by kind of
                      disagreement (different links / only luna found one /
                      only terra found one), drawn with SEED so it is the same
                      sample every run

Both modes write to the same annotations file, so anything labelled in the
sample counts towards the full run: running the full mode afterwards only asks
for the restaurants that are not annotated yet.

Progress is saved to disk after every single annotation, so you can quit any
time (Ctrl+C or 'q') and re-run the script later (in the same mode) to pick up
exactly where you left off.

Author: @Claude Code
Reviewed: @Rakul M. H. Tórgarð

Run from the terminal:
    python annotate_menulinks.py            # full dataset
    python annotate_menulinks.py --sample   # 10% sample
"""

import argparse
import sys

import numpy as np
import pandas as pd
from config import API_DATA_DIR

SOURCE_PATH = API_DATA_DIR / "full_dataset_disagreement_links_to_annotate.csv"
ANNOTATIONS_PATH = API_DATA_DIR / "full_dataset_menulink_annotations.csv"

MODEL_COLS = ["gpt-5.6-luna", "gpt-5.6-terra"]
SEED = 42           # fixes the shuffled model order and the sample, so resuming shows the same items in the same order
SAMPLE_FRACTION = 0.10
NO_URL = "<no-url-found>"

# Case 1: the model returned a URL
URL_LABELS = {
    "1": "correct_official_website",
    "2": "correct_third_party",
    "3": "incorrect",
}
# Case 2: the model returned no URL
EMPTY_LABELS = {
    "1": "correct_no_menu_found",
    "2": "incorrect_menu_exists",
}


def load_source() -> pd.DataFrame:
    # keep_default_na=False so empty responses are read as "" instead of NaN
    return pd.read_csv(SOURCE_PATH, keep_default_na=False)


def disagreement_kind(source: pd.DataFrame) -> pd.Series:
    luna_found = source["gpt-5.6-luna"].str.strip() != ""
    terra_found = source["gpt-5.6-terra"].str.strip() != ""
    kind = pd.Series("different_links", index=source.index)
    kind[luna_found & ~terra_found] = "only_luna_found"
    kind[~luna_found & terra_found] = "only_terra_found"
    return kind


def sample_ids(source: pd.DataFrame) -> pd.Index:
    """The restaurant ids in the 10% sample, stratified by kind of disagreement. Same ids for the same file and SEED."""
    sample = source.groupby(disagreement_kind(source), group_keys=False).sample(frac=SAMPLE_FRACTION, random_state=SEED)
    return pd.Index(sample["id"])


def build_queue(sample: bool = False) -> pd.DataFrame:
    source = load_source()
    source["restaurant_order"] = np.arange(len(source))

    long = source.melt(
        id_vars=["id", "name", "googleMapsUri", "restaurant_order"],
        value_vars=MODEL_COLS,
        var_name="model",
        value_name="url",
    )
    long["url"] = long["url"].str.strip()

    # random order of the models within each restaurant, so the annotator can't tell them apart by position
    long["model_order"] = np.random.default_rng(SEED).random(len(long))
    long = long.sort_values(["restaurant_order", "model_order"])

    if sample:
        long = long[long["id"].isin(sample_ids(source))]
    return long.drop(columns=["restaurant_order", "model_order"]).reset_index(drop=True)


def load_annotations() -> pd.DataFrame:
    if ANNOTATIONS_PATH.exists():
        return pd.read_csv(ANNOTATIONS_PATH, keep_default_na=False)
    return pd.DataFrame(columns=["id", "model", "url", "label", "notes"])


def remaining_queue(queue: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    if annotations.empty:
        return queue
    done = pd.MultiIndex.from_frame(annotations[["id", "model"]])
    return queue[~pd.MultiIndex.from_frame(queue[["id", "model"]]).isin(done)]


def prompt_label(is_empty: bool) -> str:
    if is_empty:
        labels = EMPTY_LABELS
        prompt = (
            "  Case 2 (empty response): could a menu have been found? Search for max. 3 minutes:\n"
            "    1. direct menu link on Google Maps  2. website from Google Maps  3. first 3 Google results\n"
            "    (third-party menus count; behind a login = not found; photos of a printed menu or board = not a menu)\n"
            "  [1] Correct (no menu found)   [2] Incorrect (menu exists)\n"
            "  [s] skip for now              [q] save and quit\n"
            "  > "
        )
    else:
        labels = URL_LABELS
        prompt = (
            "  Case 1 (URL): is the menu of the correct restaurant visible at the URL?\n"
            "    (no clicking links; behind a login = not visible; photos of a printed menu or board = not a menu)\n"
            "  [1] Correct (official website, confident)\n"
            "  [2] Correct (third party, or unsure whether it is the official website)\n"
            "  [3] Incorrect\n"
            "  [s] skip for now   [q] save and quit\n"
            "  > "
        )
    while True:
        choice = input(prompt).strip().lower()
        if choice in labels:
            return labels[choice]
        if choice in ("s", "q"):
            return choice
        print("  Invalid input, try again.\n")


def prompt_notes(label: str) -> str:
    if label == "incorrect_menu_exists":
        # the codebook asks for the URL of the menu that was found
        while True:
            notes = input("  paste the URL of the menu you found (required): ").strip()
            if notes:
                return notes
            print("  A menu URL is required for this label.")
    return input("  notes (optional, press enter to skip): ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate the models' menu links following the Annotation Codebook.")
    parser.add_argument("--sample", action="store_true",
                        help=f"only annotate the fixed {SAMPLE_FRACTION * 100:.0f}%% sample of restaurants (seed {SEED})")   # %% escapes % for argparse
    args = parser.parse_args()

    queue = build_queue(sample=args.sample)
    annotations = load_annotations()
    remaining = remaining_queue(queue, annotations)

    mode = f"{SAMPLE_FRACTION:.0%} sample" if args.sample else "full dataset"
    n_done = len(queue) - len(remaining)   # only count annotations that belong to this mode's queue
    print(f"Mode: {mode}. {n_done}/{len(queue)} already annotated. {len(remaining)} remaining.\n")

    last_id = None

    for _, row in remaining.iterrows():
        if row["id"] != last_id:
            print("\n" + "=" * 70)
            print(f"RESTAURANT: {row['name']}")
            print(f"  Google Maps: {row['googleMapsUri'] or '(none on file)'}")
            print("=" * 70)
            last_id = row["id"]

        is_empty = row["url"] == ""
        print(f"\n[{n_done + 1}/{len(queue)}]")
        print(f"  response: {NO_URL if is_empty else row['url']}")

        result = prompt_label(is_empty)

        if result == "q":
            print(f"\nSaved. {n_done}/{len(queue)} annotated so far. Progress stored at {ANNOTATIONS_PATH}")
            sys.exit(0)
        if result == "s":
            continue

        notes = prompt_notes(result)

        annotations = pd.concat(
            [
                annotations,
                pd.DataFrame(
                    [{"id": row["id"], "model": row["model"], "url": row["url"], "label": result, "notes": notes}]
                ),
            ],
            ignore_index=True,
        )
        annotations.to_csv(ANNOTATIONS_PATH, index=False)
        n_done += 1

    print(f"\nAll entries annotated! {n_done}/{len(queue)} saved to {ANNOTATIONS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nStopped. Progress up to the last label is stored at {ANNOTATIONS_PATH}")
