"""Collect the menus of every Wolt restaurant in an area, using Wolt's API.

Steps, for one area (a rough lat/lon bounding box in AREAS):
    1. venues   ask Wolt which venues deliver to a few points spread over the
                area, and keep the venues located inside the box
    2. menus    download each venue's menu (the "assortment") by its slug
    3. items    flatten all menus into one row per menu item

Both Wolt endpoints are internal (undocumented) and rate limited, so requests
are spaced out and a 429 (Too Many Requests) makes the script wait and retry.
Query points and downloaded menus are cached, so re-running only fetches what is missing.

Output (in data/wolt/<area>/):
    points/          <lat>_<lon>.json, Wolt's raw answer per query point
    venues.csv       every Wolt venue inside the box
    assortments/     <slug>.json, the raw menu per venue
    items.csv        one row per menu item: venue, category, item, description, price in DKK

Author: @Claude Code

Run from the terminal (inside src/cph-menu-finder):
    python wolt_menus.py                  # Vesterbro (default)
    python wolt_menus.py --area vesterbro
    python wolt_menus.py --area copenhagen
"""

import argparse
import json
import math
import sys
import time

import httpx
import pandas as pd
from config import WOLT_DATA_DIR
from tqdm import tqdm

# (lat_min, lat_max, lon_min, lon_max). Rough boxes, so they overlap neighbouring areas.
AREAS = {
    "vesterbro": (55.6600, 55.6760, 12.5300, 12.5660),
    "copenhagen": (55.51, 55.82, 12.23, 12.73),   # same bounds as the google maps quadtree (epsg.io/4693-area)
}

VENUES_API = "https://consumer-api.wolt.com/v1/pages/restaurants?lat={lat:.5f}&lon={lon:.5f}"
ASSORTMENT_API = "https://consumer-api.wolt.com/consumer-api/consumer-assortment/v1/venues/slug/{slug}/assortment"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "da,en;q=0.8"}

QUERY_SPACING_KM = 2.0   # distance between the points we ask Wolt about (a 4 km grid found >= 99.8% of the venues a 2 km grid did)
REQUEST_DELAY = 3.0      # seconds between two requests to Wolt
MAX_RETRIES = 6


def get(client: httpx.Client, url: str) -> httpx.Response:
    """GET with a pause after every request, and waiting + retrying when Wolt rate limits us."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.get(url)
        except httpx.TransportError as e:   # timeouts, dropped connections
            print(f"{type(e).__name__}, retrying in 30s", file=sys.stderr)
            time.sleep(30)
            continue
        time.sleep(REQUEST_DELAY)
        if response.status_code != 429:
            return response
        wait = int(response.headers.get("retry-after") or 0) or 30 * (attempt + 1)
        print(f"rate limited, waiting {wait}s", file=sys.stderr)
        time.sleep(wait)
    return response


def query_points(box: tuple) -> list[tuple[float, float]]:
    """A grid of points QUERY_SPACING_KM apart covering the box."""
    lat_min, lat_max, lon_min, lon_max = box
    km_lat = 111.0
    km_lon = 111.0 * math.cos(math.radians((lat_min + lat_max) / 2))
    n_lat = max(1, math.ceil((lat_max - lat_min) * km_lat / QUERY_SPACING_KM))
    n_lon = max(1, math.ceil((lon_max - lon_min) * km_lon / QUERY_SPACING_KM))
    lats = [lat_min + (i + 0.5) * (lat_max - lat_min) / n_lat for i in range(n_lat)]
    lons = [lon_min + (j + 0.5) * (lon_max - lon_min) / n_lon for j in range(n_lon)]
    return [(lat, lon) for lat in lats for lon in lons]


def fetch_venues(client: httpx.Client, box: tuple, points_dir) -> pd.DataFrame:
    """Every Wolt venue that delivers to at least one query point and is located inside the box."""
    points_dir.mkdir(parents=True, exist_ok=True)
    venues = {}
    for lat, lon in tqdm(query_points(box), desc="venues"):
        cache = points_dir / f"{lat:.5f}_{lon:.5f}.json"
        if not cache.exists():
            response = get(client, VENUES_API.format(lat=lat, lon=lon))
            response.raise_for_status()
            cache.write_bytes(response.content)
        for section in json.loads(cache.read_bytes()).get("sections", []):
            for item in section.get("items", []):
                v = item.get("venue")
                if v:
                    venues[v["id"]] = {
                        "wolt_id": v["id"],
                        "name": v["name"],
                        "slug": v["slug"],
                        "address": v.get("address", ""),
                        "lat": v["location"][1],
                        "lon": v["location"][0],
                    }
    venues = pd.DataFrame(venues.values())
    lat_min, lat_max, lon_min, lon_max = box
    inside = venues["lat"].between(lat_min, lat_max) & venues["lon"].between(lon_min, lon_max)
    return venues[inside].sort_values("slug").reset_index(drop=True)


def fetch_assortments(client: httpx.Client, slugs: list[str], out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in slugs if not (out_dir / f"{s}.json").exists()]
    print(f"menus: {len(todo)} to download ({len(slugs) - len(todo)} cached)")
    for slug in tqdm(todo, desc="menus"):
        response = get(client, ASSORTMENT_API.format(slug=slug))
        if response.status_code == 200:
            (out_dir / f"{slug}.json").write_bytes(response.content)
        else:
            print(f"{slug}: HTTP {response.status_code}", file=sys.stderr)


def category_of_items(categories: list[dict], parent: str = "") -> dict[str, str]:
    """item id -> category name, walking into subcategories ('Parent / Sub')."""
    lookup = {}
    for c in categories:
        name = f"{parent} / {c['name']}" if parent else c["name"]
        for item_id in c.get("item_ids", []):
            lookup.setdefault(item_id, name)
        for item_id, sub in category_of_items(c.get("subcategories", []), name).items():
            lookup.setdefault(item_id, sub)
    return lookup


def flatten(venues: pd.DataFrame, assortment_dir) -> pd.DataFrame:
    rows = []
    for v in venues.itertuples():
        path = assortment_dir / f"{v.slug}.json"
        if not path.exists():
            continue
        assortment = json.loads(path.read_text())
        categories = category_of_items(assortment.get("categories", []))
        for item in assortment.get("items", []):
            price = item.get("price")
            rows.append({
                "slug": v.slug,
                "venue": v.name,
                "category": categories.get(item["id"], ""),
                "item": item.get("name", ""),
                "description": item.get("description", ""),
                "price_dkk": price / 100 if price is not None else None,   # Wolt prices are in øre
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--area", choices=AREAS, default="vesterbro")
    args = parser.parse_args()
    box = AREAS[args.area]
    out_dir = WOLT_DATA_DIR / args.area
    out_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(headers=HEADERS, timeout=30) as client:
        venues = fetch_venues(client, box, out_dir / "points")
        venues.to_csv(out_dir / "venues.csv", index=False)
        print(f"{len(venues)} Wolt venues in {args.area}")
        fetch_assortments(client, list(venues["slug"]), out_dir / "assortments")

    items = flatten(venues, out_dir / "assortments")
    items.to_csv(out_dir / "items.csv", index=False)
    print(f"{len(items)} menu items from {items['slug'].nunique()} venues -> {out_dir / 'items.csv'}")


if __name__ == "__main__":
    main()
