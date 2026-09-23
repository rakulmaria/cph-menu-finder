"""Manual annotation tool for menu link quality.

Goes through every (restaurant, model, url) entry in
filtered_types_manual_inspection.csv one at a time and asks you to
categorize the link. Progress is saved to disk after every single
annotation, so you can quit any time (Ctrl+C or 'q') and re-run the
script later to pick up exactly where you left off.

Run from the terminal:
    python annotate_menulinks.py
"""

import sys

import pandas as pd

from config import RUNS_DATA_DIR

SOURCE_PATH = RUNS_DATA_DIR / "filtered_types_manual_inspection.csv"
ANNOTATIONS_PATH = RUNS_DATA_DIR / "menulink_annotations.csv"

MODEL_COLS = ["claude-sonnet-5", "gemini-3.7-flash", "gpt-5.6-luna", "gpt-5.6-terra"]

CATEGORIES = {
    "1": "correct_website",
    "2": "correct_thirdparty",
    "3": "broken_link",
    "4": "website_not_menu",
    "5": "incorrect",
    "6": "other",
}


def build_queue() -> pd.DataFrame:
    source = pd.read_csv(SOURCE_PATH, index_col=0)
    long = source.melt(
        id_vars=["id", "name", "websiteUri", "googleMapsUri"],
        value_vars=MODEL_COLS,
        var_name="model",
        value_name="url",
    )
    long = long.dropna(subset=["url"])
    long = long[long["url"].str.strip() != ""]
    return long.sort_values(["id", "model"]).reset_index(drop=True)


def load_annotations() -> pd.DataFrame:
    if ANNOTATIONS_PATH.exists():
        return pd.read_csv(ANNOTATIONS_PATH)
    return pd.DataFrame(columns=["id", "model", "url", "category", "notes"])


def remaining_queue(queue: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    if annotations.empty:
        return queue
    done = pd.MultiIndex.from_frame(annotations[["id", "model"]])
    return queue.set_index(["id", "model"]).drop(index=done, errors="ignore").reset_index()


def prompt_category() -> str:
    prompt = (
        "  [1] correct_website   [2] correct_thirdparty   [3] broken_link\n"
        "  [4] website_not_menu  [5] incorrect            [6] other\n"
        "  [s] skip for now      [q] save and quit\n"
        "  > "
    )
    while True:
        choice = input(prompt).strip().lower()
        if choice in CATEGORIES:
            return CATEGORIES[choice]
        if choice in ("s", "q"):
            return choice
        print("  Invalid input, try again.\n")


def main() -> None:
    queue = build_queue()
    annotations = load_annotations()
    remaining = remaining_queue(queue, annotations)

    print(f"{len(annotations)}/{len(queue)} already annotated. {len(remaining)} remaining.\n")

    last_id = None
    n_done = len(annotations)

    for _, row in remaining.iterrows():
        if row["id"] != last_id:
            print("\n" + "=" * 70)
            print(f"NEW RESTAURANT: {row['name']}")
            print(f"  reference website: {row['websiteUri'] if pd.notna(row['websiteUri']) else '(none on file)'}")
            print(f"  reference googleMapsURI: {row['googleMapsUri'] if pd.notna(row['googleMapsUri']) else '(none on file)'}")
            print("=" * 70)
            last_id = row["id"]

        print(f"\n[{n_done + 1}/{len(queue)}] model: {row['model']}")
        print(f"  found url: {row['url']}")

        result = prompt_category()

        if result == "q":
            print(f"\nSaved. {n_done}/{len(queue)} annotated so far. Progress stored at {ANNOTATIONS_PATH}")
            sys.exit(0)
        if result == "s":
            continue

        notes = input("  notes (optional, press enter to skip): ").strip()

        annotations = pd.concat(
            [
                annotations,
                pd.DataFrame(
                    [{"id": row["id"], "model": row["model"], "url": row["url"], "category": result, "notes": notes}]
                ),
            ],
            ignore_index=True,
        )
        annotations.to_csv(ANNOTATIONS_PATH, index=False)
        n_done += 1

    print(f"\nAll entries annotated! {n_done}/{len(queue)} saved to {ANNOTATIONS_PATH}")


if __name__ == "__main__":
    main()
