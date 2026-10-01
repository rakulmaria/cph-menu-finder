"""

@author: Rakul Tórgarð
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
