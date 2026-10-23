"""Download the menu pages that the models suggested, so the menus can be analysed offline.

Reads the menuLink from every answer in the two final response files (terra and
luna), de-duplicates the URLs, and fetches each unique URL once in two stages:

    1. http     plain HTTP fetch (httpx) of every URL. Wolt URLs are fetched from
                Wolt's JSON menu API instead of the page, which gives the items
                and prices directly. HTML pages with almost no visible text (the
                page is built by JavaScript, or a bot challenge was shown) are
                marked needs_js, and 401/403/429/503 responses are marked blocked.
    2. browser  re-fetches the needs_js/blocked URLs with a real browser
                (Playwright + Chromium) and saves the rendered DOM.

robots.txt is respected per site. A robots.txt that can't be read (missing, or
hidden behind a bot challenge) counts as allowing everything.

Output (in data/menu_pages/):
    menu_urls.csv   one row per (restaurant, model) with a URL, plus the url_hash
                    that links it to the fetch log and the files
    fetch_log.csv   one row per fetch attempt (a URL can have both an http and a
                    browser row), with status, content type, method and outcome
    pages/          <url_hash>.<ext> from stage 1 (.html/.pdf/.jpg/.json, ...)
                    <url_hash>.rendered.html from stage 2

The fetch log is appended to after every URL, so the script can be stopped any
time and re-run to continue where it left off. Failed URLs are not retried
unless --retry-failed is given.

@author: Rakul Tórgarð & Claude Code
@review: Rakul Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    python fetch_menus.py                     # run script
    python fetch_menus.py --limit 50          # test run on the first 50 URLs
    python fetch_menus.py --retry-failed      # try failed URLs again
    python fetch_menus.py --headed            # show the browser window

First time only: `uv run playwright install chromium`
"""

import argparse
import asyncio
import csv
import hashlib
import json
import mimetypes
import re
import time
from playwright.async_api import async_playwright
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
import pandas as pd
from config import API_DATA_DIR, MENU_PAGES_DIR
from tqdm import tqdm

RESPONSE_FILES = {
    "gpt-5.6-terra": API_DATA_DIR / "responses" / "20261001-133048-gpt-5.6-terra.csv",
    "gpt-5.6-luna": API_DATA_DIR / "responses" / "20261002-094032-gpt-5.6-luna.csv",
}

PAGES_DIR = MENU_PAGES_DIR / "pages"
MANIFEST_PATH = MENU_PAGES_DIR / "menu_urls.csv"
LOG_PATH = MENU_PAGES_DIR / "fetch_log.csv"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)

ROBOTS_AGENT = "cph-menu-finder"  # name checked against robots.txt rules
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "da,en;q=0.8"}

HTTP_CONCURRENCY = 16       # parallel requests in total
BROWSER_CONCURRENCY = 3     # parallel browser tabs
DOMAIN_DELAY = 1.0          # seconds between two requests to the same domain
HTTP_TIMEOUT = 30           # seconds
BROWSER_TIMEOUT = 45_000    # ms
MIN_TEXT_CHARS = 200        # an HTML page with less visible text than this needs JavaScript

BLOCKED_STATUSES = {401, 403, 429, 503}
MAGIC_BYTES = {b"%PDF": "application/pdf", b"\xff\xd8\xff": "image/jpeg", b"\x89PNG": "image/png", b"RIFF": "image/webp"}

LOG_COLUMNS = [
    "url_hash", "url", "domain", "method", "final_url", "status_code", "content_type",
    "file", "bytes", "text_chars", "n_items", "outcome", "error", "fetched_at",
]


def url_hash(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def build_manifest() -> pd.DataFrame:
    """One row per (restaurant, model) whose answer has a menuLink."""
    rows = []
    for model, path in RESPONSE_FILES.items():
        for answer in pd.read_csv(path)["answer"]:
            answer = json.loads(answer)
            url = (answer.get("menuLink") or "").strip()
            if url.startswith(("http://", "https://")):
                rows.append({"restaurant_id": answer["id"], "model": model, "url": url, "format": answer.get("format", "")})
    manifest = pd.DataFrame(rows)
    manifest["url_hash"] = manifest["url"].map(url_hash)
    return manifest


def load_log() -> pd.DataFrame:
    if LOG_PATH.exists():
        return pd.read_csv(LOG_PATH, keep_default_na=False)
    return pd.DataFrame(columns=LOG_COLUMNS)


def append_log(row: dict) -> None:
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in LOG_COLUMNS})


def visible_text(html: str) -> str:
    html = re.sub(r"<(script|style|noscript|template)\b.*?</\1>", " ", html, flags=re.S | re.I)
    return " ".join(re.sub(r"<[^>]+>", " ", html).split())


def looks_like_challenge(text: str) -> bool:
    return len(text) < 1000 and any(marker in text.lower() for marker in ["just a moment", "enable javascript", "checking your browser", "verify you are human"])


def extension(content_type: str, content: bytes) -> str:
    mime = content_type.split(";")[0].strip().lower()
    if mime in ("", "application/octet-stream", "binary/octet-stream"):
        # some file hosts don't say what they serve; go by the first bytes instead
        mime = next((m for magic, m in MAGIC_BYTES.items() if content.startswith(magic)), mime)
    return {"text/html": ".html", "image/jpeg": ".jpg", "text/plain": ".txt"}.get(mime) or mimetypes.guess_extension(mime) or ".bin"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Politeness:
    """Per-domain rate limit and robots.txt cache, shared by both stages."""

    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.locks: dict[str, asyncio.Lock] = {}
        self.last_request: dict[str, float] = {}
        self.robots: dict[str, RobotFileParser | None] = {}
        self.robots_locks: dict[str, asyncio.Lock] = {}

    async def wait_turn(self, url: str) -> None:
        """Waits until DOMAIN_DELAY has passed since the last request to this domain."""
        d = domain(url)
        async with self.locks.setdefault(d, asyncio.Lock()):
            wait = self.last_request.get(d, 0) + DOMAIN_DELAY - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self.last_request[d] = time.monotonic()

    async def allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"
        async with self.robots_locks.setdefault(origin, asyncio.Lock()):
            if origin not in self.robots:
                self.robots[origin] = await self.read_robots(origin)
        parser = self.robots[origin]
        return parser is None or parser.can_fetch(ROBOTS_AGENT, url)

    async def read_robots(self, origin: str) -> RobotFileParser | None:
        """None (= allow everything) when there is no readable robots.txt."""
        try:
            await self.wait_turn(origin)
            response = await self.client.get(f"{origin}/robots.txt")
        except httpx.HTTPError:
            return None
        if response.status_code != 200 or "text/plain" not in response.headers.get("content-type", ""):
            return None
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser


# ---------- stage 1: plain HTTP + Wolt API ----------

async def fetch_wolt(url: str, slug: str, client: httpx.AsyncClient, polite: Politeness) -> dict:
    api_url = f"https://consumer-api.wolt.com/consumer-api/consumer-assortment/v1/venues/slug/{slug}/assortment"
    row = {"method": "wolt_api", "final_url": api_url}
    await polite.wait_turn(api_url)
    response = await client.get(api_url)
    row.update(status_code=response.status_code, content_type=response.headers.get("content-type", ""))
    if response.status_code != 200:
        row["outcome"] = "http_error"
        return row
    path = PAGES_DIR / f"{url_hash(url)}.json"
    path.write_bytes(response.content)
    n_items = len(response.json().get("items", []))
    row.update(file=path.name, bytes=len(response.content), n_items=n_items, outcome="ok" if n_items else "empty")
    return row


async def fetch_http(url: str, client: httpx.AsyncClient, insecure_client: httpx.AsyncClient, polite: Politeness) -> dict:
    row = {"method": "http"}
    await polite.wait_turn(url)
    try:
        response = await client.get(url)
    except httpx.ConnectError as e:
        if "CERTIFICATE" not in str(e).upper():
            raise
        # some restaurant sites have broken TLS certificates; fetch anyway and note it
        response = await insecure_client.get(url)
        row["error"] = "ssl_certificate_not_verified"
    content_type = response.headers.get("content-type", "")
    row.update(final_url=str(response.url), status_code=response.status_code, content_type=content_type)

    if response.status_code in BLOCKED_STATUSES:
        row["outcome"] = "blocked"
        return row
    if response.status_code >= 400:
        row["outcome"] = "http_error"
        return row

    path = PAGES_DIR / f"{url_hash(url)}{extension(content_type, response.content)}"
    path.write_bytes(response.content)
    row.update(file=path.name, bytes=len(response.content), outcome="ok")
    if "html" in content_type.lower():
        text = visible_text(response.text)
        row["text_chars"] = len(text)
        if len(text) < MIN_TEXT_CHARS or looks_like_challenge(text):
            row["outcome"] = "needs_js"
    return row


async def run_http_stage(urls: list[str]) -> None:
    wolt_slug = re.compile(r"wolt\.com/.+/(?:restaurant|venue)/([^/?#]+)")
    semaphore = asyncio.Semaphore(HTTP_CONCURRENCY)
    timeout = httpx.Timeout(HTTP_TIMEOUT)
    async with (
        httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=timeout) as client,
        httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=timeout, verify=False) as insecure_client,
    ):
        polite = Politeness(client)
        progress = tqdm(total=len(urls), desc="http")

        async def one(url: str) -> None:
            async with semaphore:
                row = {"url_hash": url_hash(url), "url": url, "domain": domain(url)}
                try:
                    if not await polite.allowed(url):
                        row.update(method="http", outcome="robots_disallowed")
                    elif slug := wolt_slug.search(url):
                        row.update(await fetch_wolt(url, slug.group(1), client, polite))
                    else:
                        row.update(await fetch_http(url, client, insecure_client, polite))
                except Exception as e:
                    row.setdefault("method", "http")
                    row.update(outcome="error", error=f"{type(e).__name__}: {e}"[:300])
                row["fetched_at"] = now()
                append_log(row)
                progress.update()

        await asyncio.gather(*(one(url) for url in urls))
        progress.close()


# ---------- stage 2: browser, for JS sites ----------

async def fetch_browser(url: str, context, polite: Politeness) -> dict:
    row = {"method": "browser"}
    page = await context.new_page()
    try:
        await polite.wait_turn(url)
        response = await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT)
        try:
            await page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass  # some pages never go fully idle (chat widgets, trackers); use what has loaded
        text = " ".join((await page.inner_text("body")).split())
        if looks_like_challenge(text):
            # bot challenges that don't need a click usually clear after a few seconds
            await page.wait_for_timeout(8_000)
            text = " ".join((await page.inner_text("body")).split())
        html = await page.content()
        row.update(
            final_url=page.url,
            status_code=response.status if response else "",
            content_type=response.headers.get("content-type", "") if response else "",
            text_chars=len(text),
        )
        path = PAGES_DIR / f"{url_hash(url)}.rendered.html"
        path.write_text(html)
        row.update(file=path.name, bytes=len(html.encode()))
        if looks_like_challenge(text) or (row["status_code"] in BLOCKED_STATUSES and len(text) < MIN_TEXT_CHARS):
            row["outcome"] = "blocked"
        elif len(text) < MIN_TEXT_CHARS:
            row["outcome"] = "empty"
        else:
            row["outcome"] = "ok"
    finally:
        await page.close()
    return row


async def run_browser_stage(urls: list[str], headed: bool) -> None:
    semaphore = asyncio.Semaphore(BROWSER_CONCURRENCY)
    async with (
        httpx.AsyncClient(headers=HEADERS, follow_redirects=True, timeout=HTTP_TIMEOUT) as client,
        async_playwright() as p,
    ):
        browser = await p.chromium.launch(headless=not headed)
        context = await browser.new_context(user_agent=USER_AGENT, locale="da-DK")
        polite = Politeness(client)
        progress = tqdm(total=len(urls), desc="browser")

        async def one(url: str) -> None:
            async with semaphore:
                row = {"url_hash": url_hash(url), "url": url, "domain": domain(url)}
                try:
                    row.update(await fetch_browser(url, context, polite))
                except Exception as e:
                    row.update(method="browser", outcome="error", error=f"{type(e).__name__}: {e}"[:300])
                row["fetched_at"] = now()
                append_log(row)
                progress.update()

        await asyncio.gather(*(one(url) for url in urls))
        progress.close()
        await browser.close()


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only fetch the first N unique URLs (for test runs)")
    parser.add_argument("--retry-failed", action="store_true", help="fetch URLs again whose last attempt did not succeed")
    parser.add_argument("--headed", action="store_true", help="show the browser window in the browser stage")
    args = parser.parse_args()

    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    manifest.to_csv(MANIFEST_PATH, index=False)
    urls = list(dict.fromkeys(manifest["url"]))[: args.limit]
    print(f"{len(manifest)} (restaurant, model) links -> {len(urls)} unique URLs")

    # run http
    log = load_log()
    done = log[log["method"].isin(["http", "wolt_api"])]
    if args.retry_failed:
        done = done[done["outcome"].isin(["ok", "needs_js", "blocked", "robots_disallowed"])]
    todo = [u for u in urls if u not in set(done["url"])]
    print(f"http stage: {len(todo)} URLs to fetch ({len(urls) - len(todo)} already done)")
    asyncio.run(run_http_stage(todo))

    # run browser
    log = load_log()
    latest_http = log[log["method"] == "http"].drop_duplicates("url", keep="last")
    candidates = set(latest_http.loc[latest_http["outcome"].isin(["needs_js", "blocked"]), "url"])
    browser_rows = log[log["method"] == "browser"]
    if args.retry_failed:
        browser_rows = browser_rows[browser_rows["outcome"] == "ok"]
    todo = [u for u in urls if u in candidates and u not in set(browser_rows["url"])]
    print(f"browser stage: {len(todo)} URLs to render")
    asyncio.run(run_browser_stage(todo, args.headed))

    log = load_log().drop_duplicates("url", keep="last")
    print("\nlatest outcome per URL:")
    print(log["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
