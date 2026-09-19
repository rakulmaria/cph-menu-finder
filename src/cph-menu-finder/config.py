from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

PROMPTS_DIR = ROOT_DIR / "prompts"

RAW_DATA_DIR = ROOT_DIR / "data" / "raw_data"    # raw data = data from google maps api, research proj
RUNS_DATA_DIR = ROOT_DIR / "data" / "api_runs"   # api_runs = responses from llm gateway
