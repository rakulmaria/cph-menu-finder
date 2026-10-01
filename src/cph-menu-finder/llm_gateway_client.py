"""Script used to run restaurant data through LLMs via LLM Gateway.

Sends the restaurants in an input file (prepared by prepare_sample.py) to one
or more models in batches, together with a system prompt from PROMPTS_DIR, and
asks each model to find the restaurant's menu using web search.

For every model, writes to RUNS_DATA_DIR:
    responses/<timestamp>-<model>.csv        one row per restaurant, with the
                                             input and the model's parsed answer
    raw_responses/<timestamp>-<model>.jsonl  the full API response per batch,
                                             used by --resume to skip done batches

@author: Rakul Tórgarð & Claude Code
@review: Rakul Tórgarð

Usage:
    python llm_gateway_client.py
    python llm_gateway_client.py --models anthropic/claude-sonnet-5 --resume
"""

import argparse
import json
import time
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from config import *
import os
import openai

# defaults, can be overridden from the command line (see main)
RESTAURANT_FILE = "copenhagen-bounds-full-2026-10-01.json"
PROMPT = "improved-20260910.md"
BATCH_SIZE = 20
SLEEP = 0.2
TEMPERATURE = 0
MODELS = [
    "openai/gpt-5.6-terra",
    "openai/gpt-5.6-luna",
    # "google-ai-studio/gemini-3.7-flash",
    # "anthropic/claude-sonnet-5",
    # "google-ai-studio/gemini-3.8-flash",
]

class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Show argument defaults in --help, and keep the module docstring's layout."""


def get_client():
    return openai.OpenAI(
        api_key=os.getenv("LLMGATEWAY_API_KEY"),
        base_url="https://api.llmgateway.io/v1",
    )


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield i // n, seq[i:i + n]


def build_system_msg(system_prompt):
    """Cache the instructions on Anthropic models; cache_control is a no-op
    (or an error) elsewhere, so other providers get a plain string."""
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ],
    }


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


def run(restaurants, models, *, prompt=PROMPT, batch_size=BATCH_SIZE, sleep=SLEEP, temperature=TEMPERATURE, resume=False):
    system_prompt = (PROMPTS_DIR / prompt).read_text(encoding="utf-8")
    client = get_client()

    responses_dir = RUNS_DATA_DIR / "responses"
    raw_responses_dir = RUNS_DATA_DIR / "raw_responses"

    responses_dir.mkdir(parents=True, exist_ok=True)
    raw_responses_dir.mkdir(parents=True, exist_ok=True)

    # One timestamp per invocation, shared by every model in this run, so
    # all the files it produces are visibly grouped together.
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    batches = list(chunked(restaurants, batch_size))

    for model in models:
        model_slug = model.split("/")[-1]
        system_msg = build_system_msg(system_prompt)

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
                    "prompt": prompt,               # which prompt produced this
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=HelpFormatter
    )
    parser.add_argument("--input", default=RESTAURANT_FILE,
                        help="restaurant input JSON file, relative to RAW_DATA_DIR")
    parser.add_argument("--prompt", default=PROMPT,
                        help="system prompt file, relative to PROMPTS_DIR")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="models to run, space separated")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="number of restaurants per request")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="sampling temperature")
    parser.add_argument("--sleep", type=float, default=SLEEP,
                        help="seconds to wait between requests")
    parser.add_argument("--resume", action="store_true",
                        help="continue each model's most recent run (by filename timestamp) instead of "
                             "starting a new one, skipping batches already present in its raw_responses jsonl")
    args = parser.parse_args()

    restaurants = json.loads((RAW_DATA_DIR / args.input).read_text(encoding="utf-8"))

    run(restaurants, args.models, prompt=args.prompt, batch_size=args.batch_size,
        sleep=args.sleep, temperature=args.temperature, resume=args.resume)


if __name__ == "__main__":
    main()
