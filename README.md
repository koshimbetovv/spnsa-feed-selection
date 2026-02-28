# SPNSA Feed Selection (S0–S4) for AML Transaction Graphs

This repository contains research code for **feed selection strategies** for the
Shortest Paths Network Search Algorithm (**SPNSA**) and a **motif-based feed strategy (S4)**
targeting illicit-enriched investigative subgraphs in transaction networks.

## What’s inside
- **SPNSA implementation** (`spnsa_feed/spnsa.py`)
- **Trivial / degree-based feed strategies** (S0–S3) (`spnsa_feed/feeds/trivial.py`)
- **Motif-based scoring + coherent feed selection (S4)** (`spnsa_feed/feeds/motif.py`)
- **Experiment runners** (`spnsa_feed/experiments/`)
- **Data loading helpers** for:
  - IBM-AML CSV transaction logs (`spnsa_feed/data/ibm_aml.py`)
  - Elliptic classes + edgelist (`spnsa_feed/data/elliptic.py`)
- **Visualization utilities** for saving SPN connected components (`spnsa_feed/viz/components.py`)
- **Legacy notebooks** (as provided) under `notebooks/legacy/`

## Installation

Recommended: create a virtual environment, then install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
pip install -e .
```

## Data (not included)
Create a local `data/` directory and place datasets there.

Suggested layout:
```
data/
  ibm-aml/
    LI-Small_Trans.csv
    HI-Small_Trans.csv
    HI-Medium_Trans.csv
  elliptic/
    elliptic_txs_classes.csv
    elliptic_txs_edgelist.csv
```

## Quick start

### 1) Build a graph (IBM-AML)
```python
from spnsa_feed.data.ibm_aml import build_graph_from_transactions, suspicious_nodes_from_transactions
from spnsa_feed.graph.components import largest_component

G = build_graph_from_transactions("data/ibm-aml/LI-Small_Trans.csv")
G_lcc = largest_component(G, component_type="weak")

criminal_nodes = suspicious_nodes_from_transactions("data/ibm-aml/LI-Small_Trans.csv")
```

### 2) Run S4 (motif-based feed) once
```python
from spnsa_feed.experiments.motif_experiment import run_once

row = run_once(
    G_base=G_lcc,
    criminal_nodes=set(criminal_nodes),
    seed=0,
    C=300_000,
    k=200,
    centers=5,
    d_max=2,
    radius=1,
    motif_params=dict(alpha=0.2, star_ratio=15.0, s_in=0.4, s_out=0.4),
    max_in=5000,
    max_out=5000,
)
print(row)
```

### 3) Save SPN connected components
```python
from spnsa_feed.viz.components import save_spn_components
from spnsa_feed.spnsa import spnsa
from spnsa_feed.feeds.trivial import random_feed

feed = random_feed(list(G_lcc.nodes()), k=200, seed=0)
SPN, _ = spnsa(G_lcc, feed, radius=1)

save_spn_components(SPN, illicit_nodes=criminal_nodes, save_dir="outputs/spn_components")
```

## Notes on project structure
This repo uses the **`src/` layout** (recommended in modern Python packaging guides) to avoid
accidental imports from the working directory and to keep the package importable only after installation.
