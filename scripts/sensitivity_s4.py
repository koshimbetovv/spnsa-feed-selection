from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Optional

# Make repo root importable when this file is placed under scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import pandas as pd

from src.graph.components import largest_component
from src.feeds.motif import S4_motif_based_coherent
from src.spnsa import spnsa


# =========================================================
# EDIT THIS SECTION ONLY
# =========================================================
DATASET_KEY = "li-small"  # one of: elliptic, hi-small, hi-medium, li-small

# Representative paper setting
K = 200
RADIUS = 1

# Choose what to evaluate for each feed:
# - "induced": induced subgraph on the selected feed nodes
# - "spnsa": SPNSA output subgraph built from the selected feed
EVAL_MODE = "induced"

# Paper-style S4 defaults for each dataset family
DEFAULT_S4_BY_DATASET: dict[str, dict] = {
    "elliptic": dict(
        C=300_000,
        centers=5,
        d_max=2,
        motif_params=dict(
            alpha=0.2,
            beta=0.3,
            gamma=0.8,
            s_in=0.5,
            s_out=0.4,
            s_leaf=0.6,
            star_ratio=15.0,   # tau in the paper
            leaf_deg_max=2,
        ),
        max_in=500,
        max_out=500,
    ),
    "hi-small": dict(
        C=300_000,
        centers=5,
        d_max=2,
        motif_params=dict(
            alpha=0.2,
            beta=0.3,
            gamma=0.8,
            s_in=0.3,
            s_out=0.4,
            s_leaf=0.6,
            star_ratio=15.0,
            leaf_deg_max=2,
        ),
        max_in=500,
        max_out=500,
    ),
    "hi-medium": dict(
        C=300_000,
        centers=5,
        d_max=2,
        motif_params=dict(
            alpha=0.2,
            beta=0.3,
            gamma=0.8,
            s_in=0.3,
            s_out=0.4,
            s_leaf=0.6,
            star_ratio=15.0,
            leaf_deg_max=2,
        ),
        max_in=500,
        max_out=500,
    ),
    "li-small": dict(
        C=300_000,
        centers=5,
        d_max=2,
        motif_params=dict(
            alpha=0.2,
            beta=0.3,
            gamma=0.8,
            s_in=0.2,
            s_out=0.4,
            s_leaf=0.6,
            star_ratio=15.0,
            leaf_deg_max=2,
        ),
        max_in=500,
        max_out=500,
    ),
}

# One-factor-at-a-time sensitivity grid
PARAM_GRID: dict[str, list] = {
    "beta": [0.1, 0.3, 0.5],
    "gamma": [0.7, 0.8, 0.9],
    "s_leaf": [0.5, 0.6, 0.7],
    "s_in": [0.3, 0.5, 0.6],
    "centers": [3, 5, 7],
    "d_max": [1, 2, 3],
}

# Output folder relative to repo root
OUTPUT_DIR = Path("outputs/sensitivity")
SAVE_CSV = True
# =========================================================


SUPPORTED_PARAMS = {
    "alpha",
    "beta",
    "gamma",
    "s_in",
    "s_out",
    "s_leaf",
    "tau",
    "leaf_deg_max",
    "centers",
    "d_max",
    "C",
    "max_in",
    "max_out",
}

PARAM_COLUMNS = [
    "alpha",
    "beta",
    "gamma",
    "s_in",
    "s_out",
    "s_leaf",
    "tau",
    "leaf_deg_max",
    "centers",
    "d_max",
    "C",
    "max_in",
    "max_out",
]


def validate_param_grid(param_grid: dict[str, list]) -> None:
    unknown = [p for p in param_grid if p not in SUPPORTED_PARAMS]
    if unknown:
        raise ValueError(f"Unsupported sensitivity parameters: {unknown}")


def load_elliptic_labeled_nodes(classes_csv: str) -> set[str]:
    df = pd.read_csv(classes_csv, usecols=["txId", "class"], dtype=str)
    df["txId"] = df["txId"].str.strip()
    df["class"] = df["class"].str.strip()
    return set(df.loc[df["class"].isin(["1", "2"]), "txId"].tolist())


def load_graph_and_labels(dataset_key: str):
    dataset_key = dataset_key.lower()

    if dataset_key == "elliptic":
        classes_csv = Path("data/elliptic/elliptic_txs_classes.csv")
        edgelist_csv = Path("data/elliptic/elliptic_txs_edgelist.csv")
        if not classes_csv.exists():
            raise FileNotFoundError(f"Missing file: {classes_csv}")
        if not edgelist_csv.exists():
            raise FileNotFoundError(f"Missing file: {edgelist_csv}")

        from src.data.elliptic import build_graph_from_edgelist, illicit_nodes_from_classes

        G = build_graph_from_edgelist(str(edgelist_csv), directed=True)
        illicit = illicit_nodes_from_classes(str(classes_csv), illicit_label="1")
        illicit = {v for v in illicit if v in G}
        labeled = load_elliptic_labeled_nodes(str(classes_csv))
        labeled = {v for v in labeled if v in G}
        dataset_name = "Elliptic"
        return G, illicit, labeled, dataset_name

    ibm_files = {
        "hi-small": ("HI-Small_Trans.csv", "IBM-AML (HI-Small)"),
        "hi-medium": ("HI-Medium_Trans.csv", "IBM-AML (HI-Medium)"),
        "li-small": ("LI-Small_Trans.csv", "IBM-AML (LI-Small)"),
    }
    if dataset_key not in ibm_files:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    file_name, dataset_name = ibm_files[dataset_key]
    csv_path = Path("data/ibm-aml") / file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    from src.data.ibm_aml import build_graph_from_transactions, illicit_nodes_from_transactions

    G = build_graph_from_transactions(str(csv_path))
    G_lcc = largest_component(G, component_type="weak")
    illicit = illicit_nodes_from_transactions(str(csv_path))
    illicit = {v for v in illicit if v in G_lcc}
    return G_lcc, illicit, None, dataset_name


def compute_metrics(H: nx.Graph, illicit: set, labeled: Optional[set] = None) -> dict[str, float]:
    nodes = set(H.nodes())
    V = int(H.number_of_nodes())
    E = int(H.number_of_edges())
    I = int(len(nodes & illicit))
    IR = float(I / V) if V > 0 else 0.0

    out = {
        "|V|": V,
        "|E|": E,
        "|I|": I,
        "IR": IR,
    }
    if labeled is not None:
        labeled_in_H = int(len(nodes & labeled))
        LIR = float(I / labeled_in_H) if labeled_in_H > 0 else 0.0
        out["|L(H)|"] = labeled_in_H
        out["LIR"] = LIR
    return out


def get_param(s4_params: dict, param_name: str):
    if param_name == "tau":
        return s4_params["motif_params"]["star_ratio"]
    if param_name in s4_params.get("motif_params", {}):
        return s4_params["motif_params"][param_name]
    return s4_params[param_name]


def set_param(s4_params: dict, param_name: str, value) -> None:
    if param_name == "tau":
        s4_params["motif_params"]["star_ratio"] = value
    elif param_name in {"alpha", "beta", "gamma", "s_in", "s_out", "s_leaf", "leaf_deg_max"}:
        s4_params["motif_params"][param_name] = value
    elif param_name in {"centers", "d_max", "C", "max_in", "max_out"}:
        s4_params[param_name] = value
    else:
        raise ValueError(f"Unsupported parameter: {param_name}")


def flatten_s4_params(s4_params: dict) -> dict[str, float]:
    return {name: get_param(s4_params, name) for name in PARAM_COLUMNS}


def build_result_row(
    *,
    dataset_name: str,
    factor: str,
    value,
    k: int,
    radius: int,
    s4_params: dict,
    metrics: dict[str, float],
) -> dict:
    row = {
        "dataset": dataset_name,
        "factor": factor,
        "value": value,
        "k": k,
        "r": radius,
        **flatten_s4_params(s4_params),
        **metrics,
    }
    return row


def run_one_setting(
    G: nx.DiGraph,
    illicit: set,
    labeled: Optional[set],
    *,
    k: int,
    radius: int,
    s4_params: dict,
) -> dict[str, float]:
    feed = S4_motif_based_coherent(G, k=k, **s4_params)

    if EVAL_MODE == "spnsa":
        H, _ = spnsa(G, feed, radius=radius, verbose=False)
    elif EVAL_MODE == "induced":
        H = G.subgraph(feed).copy()
    else:
        raise ValueError(f"Unknown EVAL_MODE: {EVAL_MODE}")

    result = compute_metrics(H, illicit, labeled=labeled)
    result["feed_size"] = len(feed)
    return result


def build_sensitivity_tables(
    G: nx.DiGraph,
    illicit: set,
    labeled: Optional[set],
    *,
    dataset_name: str,
    k: int,
    radius: int,
    base_s4_params: dict,
    param_grid: dict[str, list],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_param_grid(param_grid)
    rows: list[dict] = []

    baseline_params = copy.deepcopy(base_s4_params)
    baseline_metrics = run_one_setting(
        G, illicit, labeled, k=k, radius=radius, s4_params=baseline_params
    )
    rows.append(
        build_result_row(
            dataset_name=dataset_name,
            factor="baseline",
            value="default",
            k=k,
            radius=radius,
            s4_params=baseline_params,
            metrics=baseline_metrics,
        )
    )

    for factor, values in param_grid.items():
        for value in values:
            s4_params = copy.deepcopy(base_s4_params)
            set_param(s4_params, factor, value)
            metrics = run_one_setting(
                G, illicit, labeled, k=k, radius=radius, s4_params=s4_params
            )
            rows.append(
                build_result_row(
                    dataset_name=dataset_name,
                    factor=factor,
                    value=value,
                    k=k,
                    radius=radius,
                    s4_params=s4_params,
                    metrics=metrics,
                )
            )

    full_df = pd.DataFrame(rows)

    ordered_cols = [
        "dataset",
        "factor",
        "value",
        "k",
        "r",
        *PARAM_COLUMNS,
        "feed_size",
        "|V|",
        "|E|",
        "|I|",
        "IR",
    ]
    if "|L(H)|" in full_df.columns:
        ordered_cols.extend(["|L(H)|", "LIR"])
    full_df = full_df[[col for col in ordered_cols if col in full_df.columns]]

    baseline_row = full_df.loc[full_df["factor"] == "baseline"].iloc[0]
    baseline_ir = float(baseline_row["IR"])
    baseline_lir = float(baseline_row["LIR"]) if "LIR" in full_df.columns else None

    summary_rows: list[dict] = []
    for factor in param_grid:
        sub = full_df.loc[full_df["factor"] == factor].copy()

        best_ir_row = sub.loc[sub["IR"].idxmax()]
        summary = {
            "dataset": dataset_name,
            "factor": factor,
            "default_value": baseline_row[factor],
            "best_value_by_IR": best_ir_row["value"],
            "best_IR": best_ir_row["IR"],
            "delta_IR_vs_default": best_ir_row["IR"] - baseline_ir,
            "best_|V|": int(best_ir_row["|V|"]),
            "best_|E|": int(best_ir_row["|E|"]),
            "best_|I|": int(best_ir_row["|I|"]),
        }

        if baseline_lir is not None and "LIR" in sub.columns:
            best_lir_row = sub.loc[sub["LIR"].idxmax()]
            summary["best_value_by_LIR"] = best_lir_row["value"]
            summary["best_LIR"] = best_lir_row["LIR"]
            summary["delta_LIR_vs_default"] = best_lir_row["LIR"] - baseline_lir

        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    return full_df, summary_df


def main() -> None:
    dataset_key = DATASET_KEY.lower()
    if dataset_key not in DEFAULT_S4_BY_DATASET:
        raise ValueError(f"Unknown DATASET_KEY: {DATASET_KEY}")

    G, illicit, labeled, dataset_name = load_graph_and_labels(dataset_key)
    base_s4_params = copy.deepcopy(DEFAULT_S4_BY_DATASET[dataset_key])

    full_df, summary_df = build_sensitivity_tables(
        G,
        illicit,
        labeled,
        dataset_name=dataset_name,
        k=K,
        radius=RADIUS,
        base_s4_params=base_s4_params,
        param_grid=PARAM_GRID,
    )

    print("\n=== S4 sensitivity analysis ===")
    print(f"Dataset: {dataset_name}")
    print(f"(k, r) = ({K}, {RADIUS})")
    print(f"Evaluation mode: {EVAL_MODE}")
    print(f"Total runs (including baseline): {len(full_df)}")
    print("\n--- Full results ---")
    print(full_df.to_string(index=False))
    print("\n--- Summary ---")
    print(summary_df.to_string(index=False))

    if SAVE_CSV:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        full_path = OUTPUT_DIR / f"s4_sensitivity_{dataset_key}_k{K}_r{RADIUS}_{EVAL_MODE}_full.csv"
        summary_path = OUTPUT_DIR / f"s4_sensitivity_{dataset_key}_k{K}_r{RADIUS}_{EVAL_MODE}_summary.csv"
        full_df.to_csv(full_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved:\n- {full_path}\n- {summary_path}")


if __name__ == "__main__":
    main()
