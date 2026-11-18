"""Export the Noise-Corrected ingredient network as a JSON file for the interactive D3 plot
in d3_interactive_plots/network.html.

The notebook writes one backbone per δ. The page lets the reader pick any δ with a slider, so
instead of one file per δ this script writes every pair once, with its z-score
(z = score / sdev_cij). backboning.thresholding keeps a pair when z > δ, so the page can do the
same filtering in the browser.

Options:
    --count item    (default) use the edge list where a pair is counted once per item
    --count venue   use the edge list where a pair is counted once per venue
                    (prepare_network.py --count venue)
The newest edge list of that kind in data/network/ is used. The page switches between the two
exports with its "Count pairs per" buttons.

Steps:
    1. Build the verified (item, ingredient) table, as in prepare_network.py.
    2. Run the NC backboning on the edge list and compute z for every pair.
    3. Keep the pairs with z > MIN_DELTA (the loosest δ the slider can show).
    4. For every whole-number δ on the slider, build the backbone as a networkx graph and detect
       communities with Louvain and with Leiden (both maximise modularity, and are weighted by the
       NC score, as the notebook layout is). The
       communities of each δ are ranked by size (0 = largest), so a colour means roughly the same
       from one δ to the next. A node that is not in the backbone at that δ gets -1.
    5. Write nodes (name, category, number of items, and the community per δ for each algorithm) and
       links (source, target, nij, z).

    6. Write every pair of the full network (before backboning) to a second file, as
       [node index, node index, weight]. The page loads it only when "show full network" is switched
       on, and draws it faintly behind the backbone.

Output (in d3_interactive_plots/, <count> is item or venue):
    network-<count>.json         {"count": ..., "edges_file": ..., "categories": {...colours},
                                  "deltas": [min, max], "nodes": [...], "links": [...]}
    network_full-<count>.json    {"links": [[i, j, weight], ...]}, i and j index into "nodes" above

@author: Rakul Tórgarð & Claude Code
@review: Rakul Tórgarð

Run from the terminal (inside src/cph-menu-finder):
    uv run python export_network_json.py
    uv run python export_network_json.py --count venue
Then serve the folder and open the page:
    cd ../../d3_interactive_plots && python3 -m http.server 8000
"""

import argparse
import json
import sys

import backboning
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from annotate_ingredient_categories import CATEGORIES
from config import *
from prepare_network import read_data

MIN_DELTA = 1.28     # the loosest δ on the slider; pairs below it are never shown
MAX_DELTA = 200      # the highest δ on the slider
SEED = 42            # Louvain and Leiden are random, a fixed seed makes the export reproducible
# algorithm name -> function(graph) giving the communities. Leiden defaults to the CPM metric, which at
# resolution 1 gives many tiny communities; modularity is what Louvain optimises, so the two can be compared
ALGORITHMS = {
    "louvain": lambda g: nx.community.louvain_communities(g, weight="score", seed=SEED),
    "leiden": lambda g: nx.community.leiden_communities(g, weight="score", metric="modularity", seed=SEED),
}
OUTPUT_DIR = ROOT_DIR / "d3_interactive_plots"
# the edge lists prepare_network.py writes, per --count. The date in the name sorts, so the last match is the newest
EDGE_PATTERNS = {"item": "edges-????-??-??.csv", "venue": "edges-????-??-??-venue.csv"}


def newest_edges(count):
    """The newest edge list of prepare_network.py for this count."""
    found = sorted(NETWORK_DATA_DIR.glob(EDGE_PATTERNS[count]))
    if not found:
        flag = "" if count == "item" else f" --count {count}"
        sys.exit(f"no {EDGE_PATTERNS[count]} in {NETWORK_DATA_DIR}, run prepare_network.py{flag} first")
    return found[-1]


def communities_per_delta(pairs, names, detect):
    """For each whole-number δ from 2 to MAX_DELTA, the community that `detect` finds for every
    ingredient in that backbone (ranked by size, 0 = largest), or -1 when it is not in the backbone."""
    result = {n: [] for n in names}
    for delta in range(2, MAX_DELTA + 1):
        backbone = pairs[pairs.z > delta]          # same rule as backboning.thresholding
        graph = nx.from_pandas_edgelist(backbone, "src", "trg", edge_attr="score")
        found = detect(graph)
        # biggest first; the smallest name breaks ties so the ranking does not depend on set order
        found = sorted(found, key=lambda c: (-len(c), min(c)))
        rank = {n: i for i, c in enumerate(found) for n in c}
        for n in names:
            result[n].append(rank.get(n, -1))
    return result


def main():
    parser = argparse.ArgumentParser(description="Export the NC ingredient network for the D3 page.")
    parser.add_argument("--count", choices=list(EDGE_PATTERNS), default="item",
                        help="the edge list to use: pairs counted once per item (default) or once per venue")
    args = parser.parse_args()
    edges_file = newest_edges(args.count)
    output = OUTPUT_DIR / f"network-{args.count}.json"
    output_full = OUTPUT_DIR / f"network_full-{args.count}.json"
    print(f"using {edges_file.name}")

    dishes_ingredients = read_data()
    n_items = dishes_ingredients.ingredient.value_counts()
    categories = pd.read_csv(WOLT_DATA_DIR / "copenhagen" / "categories.csv").set_index("ingredient_verified").category

    table, _, _ = backboning.read(edges_file, column_of_interest="weight",
                                  triangular_input=True, consider_self_loops=False, undirected=True, sep=",")
    nc_table = backboning.noise_corrected(table, undirected=True)

    pairs = nc_table[nc_table.src < nc_table.trg].copy()    # every pair is stored in both directions
    pairs["z"] = pairs.score / pairs.sdev_cij
    pairs = pairs.dropna(subset="z")
    pairs = pairs[pairs.z > MIN_DELTA].sort_values("z", ascending=False)

    # only ingredients that are in at least one pair; the rest could never be drawn
    names = sorted(set(pairs.src) | set(pairs.trg))
    communities = {algorithm: communities_per_delta(pairs, names, detect) for algorithm, detect in ALGORITHMS.items()}
    nodes = [{"id": n, "category": categories[n], "n": int(n_items[n]),
              "communities": {algorithm: found[n] for algorithm, found in communities.items()}} for n in names]
    links = [{"source": r.src, "target": r.trg, "nij": int(r.nij), "z": round(float(r.z), 3)}
             for r in pairs.itertuples()]

    # same colour per category as the notebook plots
    colors = {name: "#%02x%02x%02x" % tuple(int(255 * c) for c in rgb)
              for name, rgb in zip(CATEGORIES.values(), plt.cm.tab20.colors)}

    OUTPUT_DIR.mkdir(exist_ok=True)
    output.write_text(json.dumps({"count": args.count, "edges_file": edges_file.name, "categories": colors,
                                  "deltas": [2, MAX_DELTA], "nodes": nodes, "links": links}, ensure_ascii=False))
    # the full network: every pair of the edge list. Pairs with an ingredient that is not in "nodes" cannot be
    # placed, so they are left out (none at the moment: all ingredients are in the backbone at the loosest δ)
    index = {n: i for i, n in enumerate(names)}
    full = pd.read_csv(edges_file)
    full = full[full.src.isin(index) & full.trg.isin(index)]
    output_full.write_text(json.dumps({"links": [[index[r.src], index[r.trg], int(r.weight)] for r in full.itertuples()]}))
    print(f"wrote {len(full)} pairs of the full network to {output_full}")
    print(f"wrote {len(nodes)} nodes and {len(links)} links to {output}")


if __name__ == "__main__":
    main()
