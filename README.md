# Copenhagen Menu Finder

*MSc thesis project: building an open, ingredient-level picture of what Copenhagen's restaurants serve.*

## Vision

Restaurant menus are a large, public description of what a city eats, but the data is scattered. Menus sit on restaurant websites, in PDFs and images, on delivery platforms, or nowhere at all. Nobody collects them in a structured form, so it is hard to ask questions like *"which ingredients dominate Copenhagen's restaurant food?"* or *"what would it take to estimate the nutritional or climate footprint of eating out?"*

This project builds that dataset in three steps:

1. **Find the menus.** For every restaurant in Copenhagen, find out whether a digital menu exists and where it is.
2. **Collect the menus.** Download the menus so they can be analysed offline.
3. **Break dishes into ingredients.** Turn free-text dish descriptions into normalised, verified ingredient lists. These can later be linked to reference data such as [DTU's Food Database](https://fcdb.fooddata.dk/search).

The project also studies the method itself: how reliably LLMs with web search can locate menus, where they disagree, and which restaurant characteristics make a menu easy or hard to find.

## Two data tracks

| | **Track A: LLM menu finder** | **Track B: Wolt menus** |
|---|---|---|
| Source | Google Maps Places data for the Copenhagen bounds (from an earlier research project) | Wolt's internal venue and menu API |
| Coverage | ~3,125 operational, "proper" restaurants (after filtering) | ~2,080 venues delivering in Copenhagen |
| How | Two LLMs (`gpt-5.6-terra`, `gpt-5.6-luna`) with web search, via LLM Gateway | Grid of delivery-address query points, 2 km apart |
| Output | A menu URL (or none) per restaurant and model, plus the downloaded pages | Structured items with names, descriptions and prices in DKK |
| Status | Done and evaluated. Downloaded pages are still raw. | Collected. Ingredient extraction and annotation in progress. |

Track A answers *"can we find menus for all restaurants?"* Track B gives clean, structured menu text quickly, so the ingredient pipeline can be built and tested on it.

## Pipeline

```
Track A                                        Track B
───────                                        ───────
Google Maps raw places                         Wolt API
   │ prepare_data.py   (filter + format)          │ wolt_menus.py        (venues → menus → items.csv)
   ▼                                              ▼
LLM input JSON                                 items.csv
   │ llm_gateway_client.py (batched, web search)  │ raw_to_ingredient.py (spaCy da_core_news_trf lemmas)
   ▼                                              ▼
responses/<model>.csv                          item_ingredients.csv + ingredients.csv
   │ annotate_menulinks.py (manual verification)  │ annotate_ingredients.py (manual verification)
   │ fetch_menus.py        (httpx + Playwright)   ▼
   ▼                                           verified ingredient lists per dish
menu_pages/  (html / pdf / images / json)
```

## Results so far

**Menu finding (Track A, full dataset of 3,125 restaurants)**

- **Cost:** about $75 in total. Terra cost $55.79 and Luna $19.10. Web search is most of Luna's cost.
- **Agreement on whether a menu exists:** 86.6% (Cohen's κ = 0.658). Both models found a menu for 2,090 restaurants and both found none for 617.
- **Agreement on the URL:** 1,010 of the 2,090 restaurants got the exact same URL after normalising. These links are accepted as correct.
- **Manual verification** of 150 disagreement cases, following the [menu-link annotation codebook](cph-menu-finder.wiki/Annotation-Codebook-model-response.md): Terra was correct 71.3% of the time and Luna 57.3%. The errors were broken links, closed restaurants, drinks-only menus, special offers and wrong restaurants.
- **What makes a menu findable** (logistic regression, inferential): having a website is by far the strongest factor (odds ratio ≈ 4.3). More Google reviews also help (≈ 1.8× per 10× more reviews).

**Ingredients (Track B)**

- Food items are filtered from drinks and extras, then lemmatised with spaCy. The kept nouns become *raw ingredients*.
- Raw ingredients are reviewed by hand, most frequent first, following the [ingredient annotation codebook](cph-menu-finder.wiki/Annotation-Codebook-ingredients.md). Each one is kept, spelling-corrected or mapped to a canonical Danish name, or dropped.

Plots for all of the above are in [`plots/`](plots/).

## Conventions

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) and are enforced by commitlint through pre-commit.
