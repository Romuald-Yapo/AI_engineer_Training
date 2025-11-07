"""
Evaluation utilities for the Streamlit medical extraction dashboard.

This module centralizes dataset loading, agreement statistics, SQLite
persistence, and helper routines used by the pipeline evaluation tab.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Sequence
import datetime as _dt
import io
import math
import sqlite3

import numpy as np
import pandas as pd
import polars as pl
import requests


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

EVAL_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS expert_annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT NOT NULL,
    concept TEXT NOT NULL,
    annotator TEXT NOT NULL,
    label INTEGER NOT NULL,
    source TEXT,
    ingested_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE TABLE IF NOT EXISTS llm_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT NOT NULL,
    concept TEXT NOT NULL,
    llm TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    prediction INTEGER NOT NULL,
    prompt_version TEXT,
    retriever TEXT,
    confidence REAL,
    latency_ms REAL,
    source TEXT,
    ingested_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_expert_note_concept
    ON expert_annotations (note_id, concept);

CREATE INDEX IF NOT EXISTS idx_llm_note_concept
    ON llm_predictions (note_id, concept);
"""

TABLE_COLUMNS = {
    "expert_annotations": [
        "note_id",
        "concept",
        "annotator",
        "label",
        "source",
        "ingested_at",
    ],
    "llm_predictions": [
        "note_id",
        "concept",
        "llm",
        "pipeline",
        "prediction",
        "prompt_version",
        "retriever",
        "confidence",
        "latency_ms",
        "source",
        "ingested_at",
    ],
}


def ensure_database_schema(db_path: Path | str) -> None:
    """Create the evaluation SQLite database if missing."""
    db_path = Path(db_path)
    if not db_path.parent.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(EVAL_SCHEMA)


def _to_pandas(df: pl.DataFrame | pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if isinstance(df, pl.DataFrame):
        return df.to_pandas()
    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")


def append_dataframe_to_db(
    df: pl.DataFrame,
    table: str,
    db_path: Path | str,
    source_label: str | None = None,
) -> int:
    """
    Persist a Polars DataFrame into SQLite.

    Returns the number of rows written.
    """
    if df is None or df.is_empty():
        return 0

    db_path = Path(db_path)
    ensure_database_schema(db_path)

    expected_columns = TABLE_COLUMNS.get(table)
    if expected_columns is None:
        raise ValueError(f"Unknown table '{table}'")

    pdf = df.to_pandas()
    pdf = pdf.copy()
    timestamp = _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    if "source" not in pdf.columns:
        pdf["source"] = source_label
    elif source_label is not None:
        pdf["source"] = pdf["source"].fillna(source_label)
    pdf["ingested_at"] = timestamp

    for col in expected_columns:
        if col not in pdf.columns:
            pdf[col] = None
    pdf = pdf[expected_columns]

    with sqlite3.connect(db_path) as conn:
        pdf.to_sql(table, conn, if_exists="append", index=False)
    return len(pdf)


def load_table_from_db(table: str, db_path: Path | str) -> pl.DataFrame:
    """Read a table from SQLite into a Polars DataFrame."""
    db_path = Path(db_path)
    if not db_path.exists():
        return pl.DataFrame()

    query = f"SELECT * FROM {table}"
    with sqlite3.connect(db_path) as conn:
        try:
            pdf = pd.read_sql_query(query, conn)
        except Exception:
            return pl.DataFrame()
    if pdf.empty:
        return pl.DataFrame()
    return pl.from_pandas(pdf)


# ---------------------------------------------------------------------------
# Remote data fetching
# ---------------------------------------------------------------------------

def build_url(base: str, endpoint: str) -> str:
    """Safely concatenate a base URL and endpoint path."""
    base = (base or "").rstrip("/")
    endpoint = (endpoint or "").lstrip("/")
    return f"{base}/{endpoint}" if base else f"/{endpoint}" if endpoint else ""


def fetch_remote_dataset(
    url: str,
    expected_format: str = "json",
    headers: dict | None = None,
    timeout: int = 30,
) -> pl.DataFrame:
    """Download a dataset from an HTTP endpoint."""
    response = requests.get(url, headers=headers or {}, timeout=timeout)
    response.raise_for_status()

    if expected_format == "json":
        payload = response.json()
        records: list[dict]
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, list):
                records = data
            else:
                # Treat dict of scalars as single row
                records = [payload]
        else:
            raise ValueError("JSON payload must be a list or object")
        return pl.DataFrame(records) if records else pl.DataFrame()

    if expected_format == "csv":
        return pl.read_csv(io.StringIO(response.text))

    raise ValueError(f"Unsupported format: {expected_format}")


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def require_columns(df: pl.DataFrame, columns: Sequence[str], label: str = "") -> pl.DataFrame:
    """Ensure a dataframe contains required columns."""
    missing = [col for col in columns if col not in df.columns]
    if missing:
        context = f" for {label}" if label else ""
        raise ValueError(f"Missing columns{context}: {', '.join(missing)}")
    return df


def deduplicate_records(df: pl.DataFrame, subset: Sequence[str]) -> pl.DataFrame:
    """Drop duplicate rows keeping the first occurrence."""
    if df.is_empty():
        return df
    return df.unique(subset=subset, keep="first")


def load_sample_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Provide small in-memory datasets for demos."""
    expert_rows = [
        {"note_id": "N001", "concept": "diabetes", "annotator": "A1", "label": 1},
        {"note_id": "N001", "concept": "diabetes", "annotator": "A2", "label": 1},
        {"note_id": "N001", "concept": "hypertension", "annotator": "A1", "label": 0},
        {"note_id": "N002", "concept": "diabetes", "annotator": "A1", "label": 0},
        {"note_id": "N002", "concept": "diabetes", "annotator": "A2", "label": 1},
        {"note_id": "N003", "concept": "hypertension", "annotator": "A1", "label": 1},
        {"note_id": "N003", "concept": "hypertension", "annotator": "A2", "label": 1},
    ]

    llm_rows = [
        {
            "note_id": "N001",
            "concept": "diabetes",
            "llm": "mistral",
            "pipeline": "baseline",
            "prediction": 1,
            "confidence": 0.91,
            "latency_ms": 820,
        },
        {
            "note_id": "N001",
            "concept": "hypertension",
            "llm": "mistral",
            "pipeline": "baseline",
            "prediction": 0,
            "confidence": 0.62,
            "latency_ms": 780,
        },
        {
            "note_id": "N002",
            "concept": "diabetes",
            "llm": "mixtral",
            "pipeline": "baseline",
            "prediction": 1,
            "confidence": 0.55,
            "latency_ms": 910,
        },
        {
            "note_id": "N003",
            "concept": "hypertension",
            "llm": "mixtral",
            "pipeline": "baseline",
            "prediction": 1,
            "confidence": 0.87,
            "latency_ms": 990,
        },
    ]

    return pl.DataFrame(expert_rows), pl.DataFrame(llm_rows)


# ---------------------------------------------------------------------------
# Agreement metrics
# ---------------------------------------------------------------------------

def _filter_concepts(df: pl.DataFrame, concepts: Sequence[str] | None) -> pl.DataFrame:
    if df.is_empty():
        return df
    if concepts:
        return df.filter(pl.col("concept").is_in(list(concepts)))
    return df


def compute_majority_vote(expert_df: pl.DataFrame) -> pl.DataFrame:
    """Compute majority label per (note_id, concept)."""
    if expert_df.is_empty():
        return pl.DataFrame(
            {"note_id": [], "concept": [], "gold_label": [], "support": [], "is_tie": []}
        )

    pdf = expert_df.to_pandas()
    records: list[dict] = []
    for (note_id, concept), group in pdf.groupby(["note_id", "concept"]):
        counts = group["label"].value_counts()
        top = counts.iloc[0]
        top_labels = counts[counts == top].index.tolist()
        gold_label = int(top_labels[0]) if len(top_labels) == 1 else None
        records.append(
            {
                "note_id": note_id,
                "concept": concept,
                "gold_label": gold_label,
                "support": int(group.shape[0]),
                "is_tie": len(top_labels) > 1,
            }
        )
    return pl.DataFrame(records)


def compute_gold_coverage(
    expert_df: pl.DataFrame,
    gold_df: pl.DataFrame,
    concept_filter: Sequence[str] | None = None,
) -> float | None:
    """Share of note/concept pairs that have a resolved gold label."""
    if expert_df.is_empty() or gold_df.is_empty():
        return None

    expert_filtered = _filter_concepts(expert_df, concept_filter)
    gold_filtered = _filter_concepts(gold_df, concept_filter)

    if expert_filtered.is_empty():
        return None

    total_pairs = (
        expert_filtered.select(["note_id", "concept"]).unique().height
    )
    resolved = (
        gold_filtered.filter(pl.col("gold_label").is_not_null())
        .select(["note_id", "concept"])
        .unique()
        .height
    )
    if total_pairs == 0:
        return None
    return resolved / total_pairs


def _cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    labels = sorted(set(a) | set(b))
    if not labels:
        return None
    counts = Counter(zip(a, b))
    p0 = sum(counts[(lab, lab)] for lab in labels) / n
    pa = Counter(a)
    pb = Counter(b)
    pe = sum((pa[lab] / n) * (pb[lab] / n) for lab in labels)
    denominator = 1 - pe
    if math.isclose(denominator, 0.0):
        return None
    return (p0 - pe) / denominator


def compute_pairwise_kappa(
    expert_df: pl.DataFrame,
    concept_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Cohen's ק for every pair of annotators per concept."""
    filtered = _filter_concepts(expert_df, concept_filter)
    if filtered.is_empty():
        return pd.DataFrame(columns=["concept", "rater_a", "rater_b", "kappa", "n_samples"])

    pdf = filtered.to_pandas()
    records: list[dict] = []
    for concept, group in pdf.groupby("concept"):
        pivot = group.pivot_table(
            index="note_id",
            columns="annotator",
            values="label",
            aggfunc="first",
        )
        annotators = pivot.columns.tolist()
        for rater_a, rater_b in combinations(annotators, 2):
            pairs = pivot[[rater_a, rater_b]].dropna()
            n_samples = len(pairs)
            if n_samples == 0:
                continue
            kappa = _cohen_kappa(pairs[rater_a].astype(int), pairs[rater_b].astype(int))
            records.append(
                {
                    "concept": concept,
                    "rater_a": rater_a,
                    "rater_b": rater_b,
                    "kappa": kappa if kappa is not None else np.nan,
                    "n_samples": n_samples,
                }
            )
    return pd.DataFrame(records)


def _label_counts(group: pd.DataFrame) -> pd.DataFrame:
    counts = group.groupby(["note_id", "label"]).size().reset_index(name="count")
    pivot = counts.pivot(index="note_id", columns="label", values="count").fillna(0)
    return pivot


def compute_fleiss_kappa_per_concept(
    expert_df: pl.DataFrame,
    concept_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Fleiss' ק across annotators for each concept."""
    filtered = _filter_concepts(expert_df, concept_filter)
    if filtered.is_empty():
        return pd.DataFrame(columns=["concept", "fleiss_kappa", "n_items"])

    pdf = filtered.to_pandas()
    results: list[dict] = []
    for concept, group in pdf.groupby("concept"):
        counts = _label_counts(group)
        if counts.empty:
            continue
        p_i = []
        n_items = 0
        category_totals = Counter()
        for _, row in counts.iterrows():
            n_i = row.sum()
            if n_i <= 1:
                continue
            n_items += 1
            category_totals.update(row.to_dict())
            numerator = sum(val * (val - 1) for val in row.values)
            p_i.append(numerator / (n_i * (n_i - 1)))

        if n_items == 0:
            continue

        mean_p = sum(p_i) / n_items
        total_ratings = sum(category_totals.values())
        p_j = [
            total / total_ratings
            for total in category_totals.values()
            if total_ratings
        ]
        pe = sum(val ** 2 for val in p_j)
        denom = 1 - pe
        kappa = (mean_p - pe) / denom if not math.isclose(denom, 0.0) else np.nan
        results.append(
            {
                "concept": concept,
                "fleiss_kappa": kappa,
                "n_items": n_items,
            }
        )
    return pd.DataFrame(results)


def compute_krippendorff_alpha_per_concept(
    expert_df: pl.DataFrame,
    concept_filter: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Krippendorff's alpha (nominal) per concept."""
    filtered = _filter_concepts(expert_df, concept_filter)
    if filtered.is_empty():
        return pd.DataFrame(columns=["concept", "krippendorff_alpha", "n_items", "mean_raters"])

    pdf = filtered.to_pandas()
    results: list[dict] = []
    for concept, group in pdf.groupby("concept"):
        counts = _label_counts(group)
        if counts.empty:
            continue
        do = 0.0
        valid_items = 0
        mean_raters_acc = 0.0
        category_totals = Counter()

        for _, row in counts.iterrows():
            n_i = row.sum()
            if n_i <= 1:
                continue
            valid_items += 1
            mean_raters_acc += n_i
            category_totals.update(row.to_dict())
            row_values = row.values
            do += sum(val * (n_i - val) for val in row_values) / (n_i - 1)

        if valid_items == 0:
            continue

        total_ratings = sum(category_totals.values())
        de = 0.0
        for total in category_totals.values():
            de += total * (total_ratings - total)
        de /= max(total_ratings - 1, 1)

        if math.isclose(de, 0.0):
            alpha = np.nan
        else:
            alpha = 1 - (do / de)

        results.append(
            {
                "concept": concept,
                "krippendorff_alpha": alpha,
                "n_items": valid_items,
                "mean_raters": mean_raters_acc / valid_items,
            }
        )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# LLM metric preparation
# ---------------------------------------------------------------------------

def prepare_llm_metrics(
    llm_df: pl.DataFrame,
    gold_df: pl.DataFrame,
    concept_filter: Sequence[str] | None,
    llm_filter: Sequence[str] | None,
    pipeline_filter: Sequence[str] | None,
) -> pd.DataFrame:
    """Join LLM predictions with gold labels and compute classic metrics."""
    if llm_df is None or llm_df.is_empty():
        return pd.DataFrame(columns=["llm", "pipeline", "concept"])
    if gold_df is None or gold_df.is_empty():
        return pd.DataFrame(columns=["llm", "pipeline", "concept"])

    llm_filtered = llm_df
    if concept_filter:
        llm_filtered = llm_filtered.filter(pl.col("concept").is_in(list(concept_filter)))
    if llm_filter:
        llm_filtered = llm_filtered.filter(pl.col("llm").is_in(list(llm_filter)))
    if pipeline_filter:
        llm_filtered = llm_filtered.filter(pl.col("pipeline").is_in(list(pipeline_filter)))

    if llm_filtered.is_empty():
        return pd.DataFrame(columns=["llm", "pipeline", "concept"])

    pred_pd = llm_filtered.to_pandas()
    gold_pd = gold_df.to_pandas()
    merged = pred_pd.merge(
        gold_pd[["note_id", "concept", "gold_label"]],
        on=["note_id", "concept"],
        how="left",
    )
    merged = merged.dropna(subset=["gold_label"])
    if merged.empty:
        return pd.DataFrame(columns=["llm", "pipeline", "concept"])

    merged["gold_label"] = merged["gold_label"].astype(int)
    merged["prediction"] = merged["prediction"].astype(int)

    opt_cols = {
        "confidence": np.nan,
        "latency_ms": np.nan,
        "prompt_version": None,
        "retriever": None,
    }
    for col, default in opt_cols.items():
        if col not in merged.columns:
            merged[col] = default

    records: list[dict] = []
    group_cols = ["llm", "pipeline", "concept"]
    for keys, group in merged.groupby(group_cols):
        tp = int(((group["prediction"] == 1) & (group["gold_label"] == 1)).sum())
        tn = int(((group["prediction"] == 0) & (group["gold_label"] == 0)).sum())
        fp = int(((group["prediction"] == 1) & (group["gold_label"] == 0)).sum())
        fn = int(((group["prediction"] == 0) & (group["gold_label"] == 1)).sum())
        support = tp + tn + fp + fn

        def safe_div(num: int, denom: int) -> float | None:
            return num / denom if denom else None

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        if precision is None or recall is None or (precision + recall) == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)

        llm_name, pipeline, concept = keys
        records.append(
            {
                "llm": llm_name,
                "pipeline": pipeline,
                "concept": concept,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "support": support,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "confidence": group["confidence"].mean(),
                "latency_ms": group["latency_ms"].mean(),
            }
        )

    return pd.DataFrame(records)
