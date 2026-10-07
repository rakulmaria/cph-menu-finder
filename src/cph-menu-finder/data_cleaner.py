"""Helper functions for loading and cleaning the data used in the analysis notebooks.

Mainly used to get the two dataframes the analysis is built on, the Google Maps
restaurants and the LLM responses, and to clean them so they can be joined and
compared.

@author: Rakul Tórgarð & Claude Code
@review: Rakul Tórgarð

"""
import json

import re
import pandas as pd
from config import *

def trim_urls(df):
    url_prefix_pattern = re.compile(r'^(?:https?://)?(?:www\.)?', re.IGNORECASE)

    df['normalized_urls'] = df['menuLink'].str.replace(url_prefix_pattern, '', regex=True)
    df['normalized_urls'] = df['normalized_urls'].str.rstrip('/')

    return df


def is_restaurant_type(types):
    """True for the Google place types that count as a "proper" restaurant type (used on primaryType)."""
    types = pd.Series(types)

    # possible "proper" restaurant types that aren't named restaurant or *_restaurant
    other_types_to_keep = ["bagel_shop", "bistro", "diner", "pizza_delivery"]
    # not considered "proper" restaurant
    discard_restaurants = ["shawarma_restaurant", "hot_dog_restaurant"]

    is_restaurant = types.str.fullmatch(r"(.+_)?restaurant", na=False) | types.isin(other_types_to_keep)
    return is_restaurant & ~types.isin(discard_restaurants)


def filter_dataframe(df):
    # remove out of order restaurants
    df = df[df["businessStatus"] == "OPERATIONAL"]

    df = df[is_restaurant_type(df["primaryType"])]

    return df

def get_df(file_path, possible_filter=None):
    """
    Mainly used to get the LLM responses as one dataframe for analysis.

    Reads the response csv files in file_path and
    - only reads the files matching possible_filter (a glob pattern, e.g. '2026100*'),
      or every csv file in the folder if no filter is given
    - unpacks the JSON answer into columns (id, name, hasFoundMenu, menuLink, format, source)
    - shortens the model name (openai/gpt-5.6-luna -> gpt-5.6-luna)
    Returns one row per restaurant per model
    """
    # only the files matching the filter (e.g. '2026100*'), otherwise every csv file in the folder
    files = sorted(file_path.glob(possible_filter if possible_filter else '*.csv'))
    cols_to_keep = ['model', 'batch_index', 'answer' ]

    dfs = []
    for f in files:
        df = pd.read_csv(f, usecols=cols_to_keep)

        # turn answer response to json and unpack it
        df['answer'] = df['answer'].apply(lambda a: json.loads(a) if isinstance(a, str) else {})
        df_answer = pd.json_normalize(df['answer'])
        df = df.drop(columns=['answer']).join(df_answer)

        df['model'] = df['model'].str.split('/').str[1]
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    return df

def get_raw_data(file='copenhagen-bounds-full-raw.json'):
    """
    Mainly used to get the full Google Maps dataframe for analysis.

    Reads the raw Google Maps places data and
    - filters by operational restaurants only
    - filters away "unproper" restaurant types (hot dog stands etc.)
    - unpacks nested fields (name, price range, location)
    Keeps the columns relevant for analysis
    """
    df = pd.read_json(RAW_DATA_DIR / file)
    df = filter_dataframe(df)
    # unpack name
    df["name"] = df["displayName"].apply(lambda d: d["text"] if isinstance(d, dict) else d)

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

    # unpack price range
    price_range = pd.json_normalize(df['priceRange']).drop(columns=['startPrice.currencyCode', 'endPrice.currencyCode'])
    price_range.rename({'startPrice.units': 'startPrice', 'endPrice.units': 'endPrice'}, axis='columns', inplace=True)
    df = pd.concat([df, price_range], axis=1)

    # unwrap location
    df["lat"] = df["location"].apply(lambda x: x["latitude"])
    df["lon"] = df["location"].apply(lambda x: x["longitude"])

    # reindex and filter columns
    df = df[cols_to_keep]

    return df
