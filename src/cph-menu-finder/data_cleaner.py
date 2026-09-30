import json

import re
import pandas as pd
from config import *

def trim_urls(df):
    url_prefix_pattern = re.compile(r'^(?:https?://)?(?:www\.)?', re.IGNORECASE)

    df['normalized_urls'] = df['menuLink'].str.replace(url_prefix_pattern, '', regex=True)
    df['normalized_urls'] = df['normalized_urls'].str.rstrip('/')

    return df

def get_df():
    df_llm = get_llm_df()
    df_sample = get_sample_df()

    return df_sample.merge(df_llm, on='id')


def get_llm_df():
    csv_path = RUNS_DATA_DIR / 'responses'
    files = sorted(csv_path.glob('20260917-*'))

    dfs = []
    for f in files:
        df_llm = pd.read_csv(f, usecols=['model', 'answer'])
        df_llm['answer'] = df_llm['answer'].apply(json.loads)
        df_answer = pd.json_normalize(df_llm['answer'])
        df_llm = df_llm.drop(columns=['answer']).join(df_answer)
        df_llm['model'] = df_llm['model'].str.split('/').str[1]
        dfs.append(df_llm)

    df_llm = pd.concat(dfs, ignore_index=True)
    df_llm = trim_urls(df_llm)
    model_menulinks = df_llm.pivot_table(index='id', columns='model', values='normalized_urls', aggfunc='first')
    
    return model_menulinks


def get_sample_df():
    df = pd.read_json(RAW_DATA_DIR / 'copenhagen-10pct-sample.raw.json')

    cols_to_keep = [
        "id", 
        "name",
        "formattedAddress", 
        "primaryType",
        "googleMapsUri",
        "websiteUri",
        "businessStatus",
        "rating", 
        "takeout",
        "delivery",
        "dineIn",
        "servesBreakfast",
        "servesLunch",
        "servesDinner",
        "servesBrunch",
        "servesDessert",
        "servesVegetarianFood",
        "priceLevel",
        "startPrice",
        "endPrice",
        "lat",
        "lon",
    ]

    # remove out of order restaurants
    df = df[df["businessStatus"] == "OPERATIONAL"]

    # unpack name
    df["name"] = df["displayName"].apply(lambda d: d["text"] if isinstance(d, dict) else d)

    # unpack price range
    price_range = pd.json_normalize(df['priceRange']).drop(columns=['startPrice.currencyCode', 'endPrice.currencyCode'])
    price_range.rename({'startPrice.units': 'startPrice', 'endPrice.units': 'endPrice'}, axis='columns', inplace=True)
    df = pd.concat([df, price_range], axis=1)

    # unwrap location
    df["lat"] = df["location"].apply(lambda x: x["latitude"])
    df["lon"] = df["location"].apply(lambda x: x["longitude"])

    primary_types_to_keep = [
        "argentinian_restaurant",
        "asian_fusion_restaurant",
        "asian_restaurant",
        "bagel_shop",
        "bistro",
        "brazilian_restaurant",
        "brunch_restaurant",
        "buffet_restaurant",
        "chicken_restaurant",
        "chinese_restaurant",
        "danish_restaurant",
        "diner",
        "ethiopian_restaurant",
        "family_restaurant",
        "fast_food_restaurant",
        "fine_dining_restaurant",
        "french_restaurant",
        "fusion_restaurant",
        "greek_restaurant",
        "halal_restaurant",
        "hamburger_restaurant",
        "hawaiian_restaurant",
        "indian_restaurant",
        "indonesian_restaurant",
        "italian_restaurant",
        "japanese_restaurant",
        "korean_restaurant",
        "mexican_restaurant",
        "pakistani_restaurant",
        "persian_restaurant",
        "pizza_delivery",
        "pizza_restaurant",
        "restaurant",
        "seafood_restaurant",
        "spanish_restaurant",
        "sushi_restaurant",
        "thai_restaurant",
        "turkish_restaurant",
        "vegetarian_restaurant",
        "vietnamese_restaurant",
    ]

    # only keep "proper" restaurants
    df = df[df["primaryType"].isin(primary_types_to_keep)]

    # reindex and filter columns
    df = df[cols_to_keep]

    return df


def prepare_df():
    df = pd.read_json(RAW_DATA_DIR / 'copenhagen-bounds-raw.json')

    # remove out of order restaurants
    df = df[df["businessStatus"] == "OPERATIONAL"]

    # unpack name
    df["name"] = df["displayName"].apply(lambda d: d["text"] if isinstance(d, dict) else d)

    # possible "proper" restaurant types that aren't named restaurant or *_restaurant
    other_types_to_keep = ["bagel_shop", "bistro", "diner", "pizza_delivery"]
    # not really restaurant
    discard_restaurants = ["shawarma_restaurant", "hot_dog_restaurant"]

    # only keep "proper" restaurants: "restaurant" or any "*_restaurant"
    is_restaurant = df["primaryType"].str.fullmatch(r"(.+_)?restaurant", na=False) | df["primaryType"].isin(other_types_to_keep)
    df = df[is_restaurant & ~df["primaryType"].isin(discard_restaurants)]
    
    cols_to_keep = [
        "id", 
        "name",
        "formattedAddress", 
        # "googleMapsUri",
        "websiteUri",
    ]
    
    # reindex and filter columns
    df = df[cols_to_keep]
    df.reset_index(drop=True, inplace=True)

    # same format as copenhagen-10pct-sample-input.json: list of objects, readable æøå and urls.
    # round-trip through to_json so missing values become null instead of NaN (invalid json)
    records = json.loads(df.to_json(orient='records'))
    
    with open(RAW_DATA_DIR / 'copenhagen-bounds-input.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
        f.write('\n')

    return df
