import json
import math
import re
from typing import Any, Dict, List, Sequence

import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


DEFAULT_SYSTEM_PROMPT = (
    "Tu es un assistant d'extraction d'information clinique. Respecte scrupuleusement la "
    "structure demandée et ne produis jamais de texte hors JSON."
)

DEFAULT_USER_HEADER = (
    "Analyse le compte-rendu suivant et indique la présence exacte des concepts fournis."
)

JSON_REMINDER = (
    "La sortie demandée est STRICTEMENT un tableau JSON "# valide de la forme 
    # "[{\"concept\": str, \"context\": str, \"presence\": str}]."
)


def _normalize_presence(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "vrai", "présent", "present", "yes", "1"}:
            return True
        if token in {"false", "faux", "absent", "no", "0"}:
            return False
        if "pas mentionne" in token or "pas mentionné" in token:
            return False
    return False


def _safe_parse_json(raw: str) -> List[Dict[str, Any]]:
    if not isinstance(raw, str):
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        pass

    match = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def _format_anti_instructions(anti: Dict[str, Any] | None) -> str:
    if not anti:
        return "Vérifie chaque affirmation et n'invente aucune information."

    fragments: List[str] = []
    if anti.get("chain_of_verification") or anti.get("chainOfVerification"):
        fragments.append(
            "Utilise une chaîne de vérification: relis ta réponse avant d'envoyer le JSON."
        )
    if anti.get("contextual_grounding") or anti.get("contextualGrounding"):
        fragments.append(
            "Chaque justification doit citer explicitement un extrait du rapport."
        )
    if anti.get("ensemble") or anti.get("ensembleSize"):
        fragments.append(
            "Reste cohérent: seules les conclusions étayées doivent être marquées comme 'True'."
        )
    if anti.get("llm_as_judge") or anti.get("llmAsJudge"):
        fragments.append("Anticipe une relecture: ne marque 'True' que si la preuve est claire.")

    if not fragments:
        fragments.append("Vérifie chaque affirmation et n'invente aucune information.")
    return " ".join(fragments)


def _build_prompt(
    system_prompt: str | None,
    user_prompt: str | None,
    anti: Dict[str, Any] | None,
) -> ChatPromptTemplate:
    header = (user_prompt or DEFAULT_USER_HEADER).strip()
    anti_text = _format_anti_instructions(anti)

    user_template = (
        f"{header}\n\n"
        "Concepts autorisés (exact match): {concept_list}\n\n"
        "Compte-rendu clinique:\n{report_text}\n\n"
        f"{anti_text}\n"
        f"{JSON_REMINDER}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt or DEFAULT_SYSTEM_PROMPT),
            ("user", user_template),
        ]
    )
    return prompt


def _build_chain(
    model_conf: Dict[str, Any],
    system_prompt: str | None,
    user_prompt: str | None,
    anti: Dict[str, Any] | None,
) -> Any:
    if not model_conf.get("model"):
        raise ValueError("Model identifier is required to build the evaluation chain.")

    prompt = _build_prompt(system_prompt, user_prompt, anti)

    kwargs: Dict[str, Any] = {
        "model": model_conf["model"],
        "temperature": float(model_conf.get("temperature", 0.0)),
    }
    for field in ("top_p", "top_k", "num_ctx"):
        if model_conf.get(field) is not None:
            kwargs[field] = model_conf[field]
    if model_conf.get("max_output") is not None:
        kwargs["num_predict"] = model_conf["max_output"]

    llm = ChatOllama(**kwargs)
    chain = prompt | llm | StrOutputParser()
    return chain.with_retry(stop_after_attempt=2, wait_exponential_jitter=True)


def _prepare_inputs(
    df: pd.DataFrame, text_col: str, concept_cols: Sequence[str]
) -> List[Dict[str, str]]:
    concepts = ", ".join(concept_cols)
    inputs: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        inputs.append(
            {
                "concept_list": concepts,
                "report_text": str(row.get(text_col, "")) if text_col in df.columns else "",
            }
        )
    print(concepts)
    return inputs


def run_model_projection(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    concept_cols: Sequence[str],
    model_conf: Dict[str, Any],
    system_prompt: str | None,
    user_prompt: str | None,
    anti: Dict[str, Any],
    max_concurrency: int = 4,
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in the evaluation dataframe.")

    chain = _build_chain(model_conf, system_prompt, user_prompt, anti)
    inputs = _prepare_inputs(df, text_col, concept_cols)
    raw_outputs = chain.batch(inputs, config={"max_concurrency": max_concurrency})

    projection = df[[id_col]].copy()
    if text_col in df.columns:
        projection[text_col] = df[text_col]
    for col in concept_cols:
        projection[col] = False

    for row_idx, raw in enumerate(raw_outputs):
        parsed = _safe_parse_json(raw)
        by_concept: Dict[str, bool] = {}
        for item in parsed:
            concept = item.get("concept")
            presence = _normalize_presence(item.get("presence"))
            if concept in concept_cols:
                by_concept[concept] = by_concept.get(concept, False) or presence

        for concept in concept_cols:
            projection.iat[row_idx, projection.columns.get_loc(concept)] = bool(
                by_concept.get(concept, False)
            )

    return projection


def _majority_vote(
    frames: List[pd.DataFrame],
    id_col: str,
    text_col: str | None,
    concept_cols: Sequence[str],
) -> pd.DataFrame:
    aligned = [frame.set_index(id_col) for frame in frames]
    base = pd.DataFrame(index=aligned[0].index)
    if text_col and text_col in aligned[0].columns:
        base[text_col] = aligned[0][text_col]

    for concept in concept_cols:
        votes = [df[concept].astype(int) for df in aligned]
        stacked = pd.concat(votes, axis=1)
        threshold = math.ceil(len(votes) / 2)
        base[concept] = (stacked.sum(axis=1) >= threshold).astype(bool)

    combined = base.reset_index().rename(columns={"index": id_col})
    if id_col not in combined.columns:
        combined.insert(0, id_col, base.index.tolist())
    return combined


def aggregate_predictions(
    per_model: List[pd.DataFrame],
    id_col: str,
    text_col: str | None,
    concept_cols: Sequence[str],
    anti: Dict[str, Any],
) -> pd.DataFrame:
    if not per_model:
        raise ValueError("No model predictions were generated.")

    ensemble_enabled = bool(anti.get("ensemble", anti.get("ensembleSize")))
    ensemble_size = int(
        anti.get("ensemble_size", anti.get("ensembleSize", len(per_model)))
    )
    ensemble_size = max(1, min(ensemble_size, len(per_model)))

    if ensemble_enabled and len(per_model) > 1:
        return _majority_vote(per_model[:ensemble_size], id_col, text_col, concept_cols)

    return per_model[0]


def apply_judge_filter(
    aggregated: pd.DataFrame,
    judge_projection: pd.DataFrame | None,
    id_col: str,
    concept_cols: Sequence[str],
) -> pd.DataFrame:
    if judge_projection is None:
        return aggregated

    merged = aggregated.merge(judge_projection[[id_col] + list(concept_cols)], on=id_col, suffixes=("", "_judge"))
    for concept in concept_cols:
        merged[concept] = merged[concept] & merged[f"{concept}_judge"]
        merged.drop(columns=[f"{concept}_judge"], inplace=True)
    return merged


def run_pipeline_prediction(
    df: pd.DataFrame,
    id_col: str,
    text_col: str,
    concept_cols: Sequence[str],
    model_confs: Sequence[Dict[str, Any]],
    system_prompt: str | None,
    user_prompt: str | None,
    anti: Dict[str, Any],
    judge_conf: Dict[str, Any] | None = None,
    max_concurrency: int = 4,
) -> Dict[str, Any]:
    per_model_outputs: List[pd.DataFrame] = []
    for conf in model_confs:
        per_model_outputs.append(
            run_model_projection(
                df,
                id_col,
                text_col,
                concept_cols,
                conf,
                system_prompt,
                user_prompt,
                anti,
                max_concurrency=max_concurrency,
            )
        )

    aggregated = aggregate_predictions(per_model_outputs, id_col, text_col, concept_cols, anti)

    judge_projection = None
    if anti.get("llm_as_judge") and judge_conf and judge_conf.get("model"):
        judge_projection = run_model_projection(
            df,
            id_col,
            text_col,
            concept_cols,
            judge_conf,
            system_prompt,
            user_prompt,
            anti,
            max_concurrency=max_concurrency,
        )
        aggregated = apply_judge_filter(aggregated, judge_projection, id_col, concept_cols)

    return {
        "per_model": per_model_outputs,
        "aggregated": aggregated,
        "judge": judge_projection,
    }


def compute_concept_metrics(
    gold_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    concept_cols: Sequence[str],
    id_col: str,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    merged = gold_df[[id_col] + list(concept_cols)].merge(
        pred_df[[id_col] + list(concept_cols)],
        on=id_col,
        suffixes=("_gold", "_pred"),
    )

    metrics: List[Dict[str, Any]] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "support": 0}

    for concept in concept_cols:
        gold_col = f"{concept}_gold"
        pred_col = f"{concept}_pred"
        gold_values = merged[gold_col]
        pred_values = merged[pred_col]

        valid_mask = gold_values.isin([0, 1])
        y_true = gold_values[valid_mask].astype(int)
        y_pred = pred_values[valid_mask].fillna(0).astype(int)

        if y_true.empty:
            metrics.append(
                {
                    "concept": concept,
                    "support": 0,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "accuracy": None,
                }
            )
            continue

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        support = len(y_true)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = (tp + tn) / support if support else 0.0

        metrics.append(
            {
                "concept": concept,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

        totals["tp"] += tp
        totals["tn"] += tn
        totals["fp"] += fp
        totals["fn"] += fn
        totals["support"] += support

    overall_precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    overall_recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0
    overall_f1 = (
        (2 * overall_precision * overall_recall / (overall_precision + overall_recall))
        if (overall_precision + overall_recall)
        else 0.0
    )
    overall_accuracy = (totals["tp"] + totals["tn"]) / totals["support"] if totals["support"] else 0.0

    summary = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "accuracy": overall_accuracy,
        "support": totals["support"],
    }

    return pd.DataFrame(metrics), summary
