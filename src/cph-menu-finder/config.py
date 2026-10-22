from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PROMPTS_DIR = ROOT_DIR / "prompts"

RAW_DATA_DIR = ROOT_DIR / "data" / "raw_data"    # raw data = data from google maps api, research proj
API_DATA_DIR = ROOT_DIR / "data" / "api_runs"   # api_runs = responses from llm gateway
MENU_PAGES_DIR = ROOT_DIR / "data" / "menu_pages"   # menu_pages = downloaded menu pages/pdfs/images + fetch log
WOLT_DATA_DIR = ROOT_DIR / "data" / "wolt"   # wolt = venues, raw menus and items.csv per area, from wolt_menus.py

PLOTS_DIR = ROOT_DIR / "plots"
