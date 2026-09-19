import argparse
import json
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from config import *
import os
import openai

RESTAURANT_FILE = "copenhagen-10pct-sample.json"
PROMPT = "improved-20260910.md"
BATCH_SIZE = 20
# Gemini counts thinking tokens against the same output budget as the
# visible answer, so at the default max_tokens it can burn the whole
# budget reasoning and never emit the JSON. Give it more headroom instead
# of lowering its reasoning_effort, so it still reasons like the others.
MAX_TOKENS_OVERRIDES = {
    "google-ai-studio/gemini-3.8-flash": 32000,
}


def get_client():
    return openai.OpenAI(
        api_key=os.getenv("LLMGATEWAY_API_KEY"),
        base_url="https://api.llmgateway.io/v1",
    )


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield i // n, seq[i:i + n]


def call_batch(client, model, system_msg, batch, *, temperature=0, reasoning_effort="medium"):
    """Send the cached instructions plus one batch of restaurants."""
    kwargs = dict(
        model=model,
        temperature=temperature,               # 0 = as reproducible as the model allows
        messages=[
            system_msg,                        # cached instructions
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
        ],
        tools=[
            {"type": "web_search"}
        ],
        reasoning_effort=reasoning_effort,
    )
    if model in MAX_TOKENS_OVERRIDES:
        kwargs["max_tokens"] = MAX_TOKENS_OVERRIDES[model]
    return client.chat.completions.create(**kwargs)


def parse_list(text):
    """Best-effort parse of the model's answer into a list of dicts, or None."""
    text = str(text).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text[text.find("\n") + 1:]
    try:
        data = json.loads(text)
    except Exception:
        i, j = text.find("["), text.rfind("]")
        if i == -1 or j <= i:
            return None
        try:
            data = json.loads(text[i:j + 1])
        except Exception:
            return None
    if isinstance(data, dict):
        data = [data]
    return data if isinstance(data, list) else None


def parse_restaurants(path):
    df = pd.read_json(path)

    if "displayName" in df.columns:
        df["name"] = df["displayName"].apply(lambda d: d["text"] if isinstance(d, dict) else d)

    df = df.reindex(columns=["id", "name", "formattedAddress", "websiteUri"])
    df = df.astype(object).where(df.notna(), None)   # NaN -> None, so missing fields serialize as JSON null
    return df.to_dict(orient="records")


def run(restaurants, models, *, batch_size=BATCH_SIZE, sleep=0.2, temperature=0, resume=False):
    system_prompt = (PROMPTS_DIR / PROMPT).read_text(encoding="utf-8")
    system_msg = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
    }
    client = get_client()

    responses_dir = RUNS_DATA_DIR / "responses"
    raw_responses_dir = RUNS_DATA_DIR / "raw_responses"

    responses_dir.mkdir(parents=True, exist_ok=True)
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    # One timestamp per invocation, shared by every model in this run, so
    # all the files it produces are visibly grouped together.
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    batches = list(chunked(restaurants, batch_size))
    batches = batches[:1]  # TEMP: testing cache_control, only run batch 0

    for model in models:
        model_slug = model.split("/")[-1]

        if resume:
            # Continue the most recent run for this model, if one exists,
            # instead of starting a fresh timestamped file.
            existing = sorted(raw_responses_dir.glob(f"*-{model_slug}.jsonl"))
            tag = existing[-1].stem if existing else f"{run_timestamp}-{model_slug}"
        else:
            tag = f"{run_timestamp}-{model_slug}"

        csv_path = responses_dir / f"{tag}.csv"
        raw_path = raw_responses_dir / f"{tag}.jsonl"

        done_batches = set()
        if resume and raw_path.exists():
            with raw_path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done_batches.add(json.loads(line)["batch_index"])
                    except (json.JSONDecodeError, KeyError):
                        continue  # truncated/partial last line -- redo that batch
            if done_batches:
                tqdm.write(f"  resuming {model}: {len(done_batches)}/{len(batches)} batches already done, skipping them")
        else:
            csv_path.unlink(missing_ok=True)      # fresh run -- drop any earlier output for this tag
            raw_path.unlink(missing_ok=True)

        for batch_index, batch in tqdm(batches, desc=model):
            if batch_index in done_batches:
                continue

            resp = call_batch(client, model, system_msg, batch,
                              temperature=temperature)
            answer = resp.choices[0].message.content

            parsed = parse_list(answer)
            by_id = {
                obj["id"]: json.dumps(obj, ensure_ascii=False)
                for obj in (parsed or [])
                if isinstance(obj, dict) and "id" in obj
            }

            if parsed is None:
                tqdm.write(f"  batch {batch_index}: no JSON list found; storing raw answer on every row")
            else:
                missing = [r["id"] for r in batch if r["id"] not in by_id]
                if missing:
                    tqdm.write(f"  batch {batch_index}: {len(missing)}/{len(batch)} ids missing from response")

            rows = [
                {
                    "prompt": PROMPT,               # which prompt produced this
                    "model": model,
                    "batch_index": batch_index,
                    "restaurant_id": r["id"],
                    "input": json.dumps(r, ensure_ascii=False),
                    "answer": by_id.get(r["id"], "" if parsed is not None else answer),
                }
                for r in batch
            ]
            pd.DataFrame(rows).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

            rec = resp.model_dump(warnings=False)
            rec["batch_index"] = batch_index
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            time.sleep(sleep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="continue each model's most recent run (by filename timestamp) instead of "
                             "starting a new one, skipping batches already present in its raw_responses jsonl")
    args = parser.parse_args()

    restaurants = parse_restaurants(RAW_DATA_DIR / RESTAURANT_FILE)

    models = [
        # "openai/gpt-5.6-terra",
        # "openai/gpt-5.6-luna",
        # "google-ai-studio/gemini-3.7-flash"
        "anthropic/claude-sonnet-5",
        # "google-ai-studio/gemini-3.8-flash",
    ]

    run(restaurants, models, resume=args.resume)


if __name__ == "__main__":
    main()
