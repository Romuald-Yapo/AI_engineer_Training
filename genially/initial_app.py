import sqlite3
from copy import deepcopy
from pathlib import Path
import polars as pl
import pandas as pd
import datetime as datetime
import json
import requests

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from initialize_pipeline_db import DB_FILENAME, SCHEMA, ensure_database
from display_text import display, highlight_contexts, configure_aggrid
from llm_extractor_pipeline import build_messages, llm_extractor
from evaluation_core import (
    append_dataframe_to_db,
    build_url,
    compute_fleiss_kappa_per_concept,
    compute_gold_coverage,
    compute_krippendorff_alpha_per_concept,
    compute_majority_vote,
    compute_pairwise_kappa,
    deduplicate_records,
    ensure_database_schema,
    fetch_remote_dataset,
    load_sample_data,
    load_table_from_db,
    prepare_llm_metrics,
    require_columns,
)

st.set_page_config(page_title="LLM-Based Medical Concept Extraction", layout="wide")

MODEL_DEFAULTS = {
    "name": "",
    "description": "",
    "model": "",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "num_ctx": 4096,
    "max_output": 1024,
}

PROMPT_DEFAULTS = {
    "name": "",
    "description": "",
    "type": "clinical",
    "content": "",
}

PIPELINE_DEFAULTS = {
    "name": "",
    "description": "",
    "data_source": "",
    "selected_models": [],
    "technical_prompt": None,
    "clinical_prompt": None,
    "ensemble": False,
    "ensemble_size": 1,
    "chain_of_verification": False,
    "contextual_grounding": False,
    "llm_as_judge": False,
    "judge_model": None,
}

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / DB_FILENAME

if not DB_PATH.exists():
    ensure_database(DB_PATH, SCHEMA)


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_pipelines_from_db() -> list[dict]:
    if not DB_PATH.exists():
        return []

    with get_db_connection() as conn:
        pipeline_rows = conn.execute(
            """
            SELECT
                id,
                name,
                description,
                data_source,
                technical_prompt_id,
                clinical_prompt_id,
                ensemble_enabled,
                ensemble_size,
                chain_of_verification,
                contextual_grounding,
                llm_as_judge,
                judge_model_id
            FROM pipelines
            ORDER BY id
            """
        ).fetchall()

        model_rows = conn.execute(
            """
            SELECT
                pm.pipeline_id,
                pm.model_id,
                pm.step_order
            FROM pipeline_models AS pm
            LEFT JOIN models AS m
              ON m.id = pm.model_id
            WHERE m.id IS NOT NULL
            ORDER BY pm.pipeline_id, pm.step_order
            """
        ).fetchall()

    models_by_pipeline: dict[int, list[int]] = {}
    for row in model_rows:
        models_by_pipeline.setdefault(row["pipeline_id"], []).append(row["model_id"])

    pipelines: list[dict] = []
    for row in pipeline_rows:
        anti = {
            "ensemble": bool(row["ensemble_enabled"]),
            "ensemble_size": row["ensemble_size"],
            "chain_of_verification": bool(row["chain_of_verification"]),
            "contextual_grounding": bool(row["contextual_grounding"]),
            "llm_as_judge": bool(row["llm_as_judge"]),
            "judge_model": row["judge_model_id"],
        }
        anti["ensembleSize"] = anti["ensemble_size"]
        anti["chainOfVerification"] = anti["chain_of_verification"]
        anti["contextualGrounding"] = anti["contextual_grounding"]
        anti["llmAsJudge"] = anti["llm_as_judge"]
        anti["judgeModel"] = anti["judge_model"]

        pipelines.append(
            {
                "id": row["id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "data_source": row["data_source"] or "",
                "selected_models": models_by_pipeline.get(row["id"], []),
                "technical_prompt": row["technical_prompt_id"],
                "clinical_prompt": row["clinical_prompt_id"],
                "anti_hallucination": anti,
            }
        )

    return pipelines


def load_models_from_db() -> list[dict]:
    if not DB_PATH.exists():
        return []

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                name,
                description,
                identifier,
                temperature,
                top_p,
                top_k,
                num_ctx,
                max_output
            FROM models
            ORDER BY id
            """
        ).fetchall()

    models: list[dict] = []
    for row in rows:
        models.append(
            {
                "id": row["id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "model": row["identifier"] or "",
                "temperature": float(row["temperature"]) if row["temperature"] is not None else 0.7,
                "top_p": float(row["top_p"]) if row["top_p"] is not None else 0.95,
                "top_k": int(row["top_k"]) if row["top_k"] is not None else 40,
                "num_ctx": int(row["num_ctx"]) if row["num_ctx"] is not None else 4096,
                "max_output": int(row["max_output"]) if row["max_output"] is not None else 1024,
            }
        )
    return models


def sync_models_from_db(state: st.session_state) -> None:
    try:
        models = load_models_from_db()
        state.model_db_error = ""
    except sqlite3.Error as exc:
        models = []
        state.model_db_error = f"Database error while loading models: {exc}"
    state.models = models


def persist_model_to_db(model: dict, model_id: int | None = None) -> int:
    payload = (
        model["name"],
        model.get("description", ""),
        model["model"],
        float(model.get("temperature", 0.7)),
        float(model.get("top_p", 0.95)),
        int(model.get("top_k", 40)),
        int(model.get("num_ctx", 4096)),
        int(model.get("max_output", 1024)),
    )

    with get_db_connection() as conn:
        if model_id is None:
            cursor = conn.execute(
                """
                INSERT INTO models (
                    name,
                    description,
                    identifier,
                    temperature,
                    top_p,
                    top_k,
                    num_ctx,
                    max_output
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            model_id = cursor.lastrowid
        else:
            conn.execute(
                """
                UPDATE models
                SET
                    name = ?,
                    description = ?,
                    identifier = ?,
                    temperature = ?,
                    top_p = ?,
                    top_k = ?,
                    num_ctx = ?,
                    max_output = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                payload + (model_id,),
            )
    return model_id


def delete_model_from_db(model_id: int) -> None:
    with get_db_connection() as conn:
        conn.execute("DELETE FROM pipeline_models WHERE model_id = ?", (model_id,))
        conn.execute(
            """
            UPDATE pipelines
            SET
                judge_model_id = NULL,
                llm_as_judge = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE judge_model_id = ?
            """,
            (model_id,),
        )
        conn.execute("DELETE FROM models WHERE id = ?", (model_id,))


def load_prompts_from_db() -> list[dict]:
    if not DB_PATH.exists():
        return []

    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                name,
                description,
                type,
                content
            FROM prompts
            ORDER BY id
            """
        ).fetchall()

    prompts: list[dict] = []
    for row in rows:
        prompts.append(
            {
                "id": row["id"],
                "name": row["name"] or "",
                "description": row["description"] or "",
                "type": row["type"],
                "content": row["content"] or "",
            }
        )
    return prompts


def sync_prompts_from_db(state: st.session_state) -> None:
    try:
        prompts = load_prompts_from_db()
        state.prompt_db_error = ""
    except sqlite3.Error as exc:
        prompts = []
        state.prompt_db_error = f"Database error while loading prompts: {exc}"
    state.prompts = prompts


def persist_prompt_to_db(prompt: dict, prompt_id: int | None = None) -> int:
    payload = (
        prompt["name"],
        prompt.get("description", ""),
        prompt["type"],
        prompt["content"],
    )

    with get_db_connection() as conn:
        if prompt_id is None:
            cursor = conn.execute(
                """
                INSERT INTO prompts (
                    name,
                    description,
                    type,
                    content
                ) VALUES (?, ?, ?, ?)
                """,
                payload,
            )
            prompt_id = cursor.lastrowid
        else:
            conn.execute(
                """
                UPDATE prompts
                SET
                    name = ?,
                    description = ?,
                    type = ?,
                    content = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                payload + (prompt_id,),
            )
    return prompt_id


def delete_prompt_from_db(prompt_id: int) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE pipelines
            SET
                technical_prompt_id = CASE WHEN technical_prompt_id = ? THEN NULL ELSE technical_prompt_id END,
                clinical_prompt_id = CASE WHEN clinical_prompt_id = ? THEN NULL ELSE clinical_prompt_id END,
                updated_at = CASE
                    WHEN technical_prompt_id = ? OR clinical_prompt_id = ?
                    THEN strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ELSE updated_at
                END
            WHERE technical_prompt_id = ? OR clinical_prompt_id = ?
            """,
            (prompt_id, prompt_id, prompt_id, prompt_id, prompt_id, prompt_id),
        )
        conn.execute("DELETE FROM prompts WHERE id = ?", (prompt_id,))


def sync_pipelines_from_db(state: st.session_state) -> None:
    try:
        pipelines = load_pipelines_from_db()
        state.pipeline_db_error = ""
    except sqlite3.Error as exc:
        pipelines = []
        state.pipeline_db_error = f"Database error: {exc}"

    state.pipelines = pipelines
    state.pipeline_id_counter = max((p["id"] for p in pipelines), default=0) + 1


def persist_pipeline_to_db(pipeline: dict, pipeline_id: int | None = None) -> int:
    anti = pipeline["anti_hallucination"]
    ensemble_flag = bool(anti.get("ensemble", anti.get("ensembleSize", False)))
    ensemble_size = int(
        anti.get("ensemble_size", anti.get("ensembleSize", PIPELINE_DEFAULTS["ensemble_size"]))
        or PIPELINE_DEFAULTS["ensemble_size"]
    )
    if ensemble_flag:
        ensemble_size = max(ensemble_size, 1)

    chain_flag = bool(anti.get("chain_of_verification", anti.get("chainOfVerification", False)))
    grounding_flag = bool(anti.get("contextual_grounding", anti.get("contextualGrounding", False)))
    judge_flag = bool(anti.get("llm_as_judge", anti.get("llmAsJudge", False)))
    judge_model = anti.get("judge_model", anti.get("judgeModel"))

    payload = (
        pipeline["name"],
        pipeline["description"],
        pipeline["data_source"],
        pipeline.get("technical_prompt"),
        pipeline.get("clinical_prompt"),
        int(ensemble_flag),
        ensemble_size,
        int(chain_flag),
        int(grounding_flag),
        int(judge_flag),
        judge_model,
    )

    with get_db_connection() as conn:
        if pipeline_id is None:
            cursor = conn.execute(
                """
                INSERT INTO pipelines (
                    name,
                    description,
                    data_source,
                    technical_prompt_id,
                    clinical_prompt_id,
                    ensemble_enabled,
                    ensemble_size,
                    chain_of_verification,
                    contextual_grounding,
                    llm_as_judge,
                    judge_model_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            pipeline_id = cursor.lastrowid
        else:
            conn.execute(
                """
                UPDATE pipelines
                SET
                    name = ?,
                    description = ?,
                    data_source = ?,
                    technical_prompt_id = ?,
                    clinical_prompt_id = ?,
                    ensemble_enabled = ?,
                    ensemble_size = ?,
                    chain_of_verification = ?,
                    contextual_grounding = ?,
                    llm_as_judge = ?,
                    judge_model_id = ?,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?
                """,
                payload + (pipeline_id,),
            )
            conn.execute(
                "DELETE FROM pipeline_models WHERE pipeline_id = ?",
                (pipeline_id,),
            )

        steps = [
            (pipeline_id, int(model_id), order)
            for order, model_id in enumerate(pipeline["selected_models"], start=1)
        ]
        if steps:
            conn.executemany(
                """
                INSERT INTO pipeline_models (pipeline_id, model_id, step_order)
                VALUES (?, ?, ?)
                """,
                steps,
            )

    return pipeline_id


def delete_pipeline_from_db(pipeline_id: int) -> None:
    with get_db_connection() as conn:
        conn.execute("DELETE FROM pipelines WHERE id = ?", (pipeline_id,))


def count_enabled_strategies(anti: dict) -> int:
    strategy_keys = (
        ("ensemble", "ensembleSize"),
        ("chain_of_verification", "chainOfVerification"),
        ("contextual_grounding", "contextualGrounding"),
        ("llm_as_judge", "llmAsJudge"),
    )
    return sum(1 for snake, camel in strategy_keys if anti.get(snake) or anti.get(camel))

def reset_model_form_fields(state: st.session_state, data: dict | None = None) -> None:
    payload = {**MODEL_DEFAULTS, **(data or {})}
    if data:
        legacy_max_tokens = data.get("max_tokens")
        if legacy_max_tokens is not None and not data.get("max_output"):
            payload["max_output"] = legacy_max_tokens
    state.model_form_name = payload["name"]
    state.model_form_description = payload["description"]
    state.model_form_model = payload["model"]
    state.model_form_temperature = float(payload["temperature"])
    state.model_form_top_p = float(payload["top_p"])
    state.model_form_top_k = int(payload["top_k"])
    state.model_form_num_ctx = int(payload["num_ctx"])
    state.model_form_max_output = int(payload["max_output"])


def reset_prompt_form_fields(state: st.session_state, data: dict | None = None) -> None:
    payload = {**PROMPT_DEFAULTS, **(data or {})}
    state.prompt_form_name = payload["name"]
    state.prompt_form_description = payload["description"]
    state.prompt_form_type = payload["type"]
    state.prompt_form_content = payload["content"]


def reset_pipeline_form_fields(state: st.session_state, data: dict | None = None) -> None:
    payload = PIPELINE_DEFAULTS.copy()
    if data:
        payload.update({
            "name": data.get("name", ""),
            "description": data.get("description", ""),
            "data_source": data.get("data_source", ""),
            "selected_models": list(data.get("selected_models", [])),
            "technical_prompt": data.get("technical_prompt"),
            "clinical_prompt": data.get("clinical_prompt"),
        })
        anti = data.get("anti_hallucination", {})
        payload["ensemble"] = bool(anti.get("ensemble", payload["ensemble"]))
        payload["ensemble_size"] = int(anti.get("ensemble_size", payload["ensemble_size"]))
        payload["chain_of_verification"] = bool(anti.get("chain_of_verification", payload["chain_of_verification"]))
        payload["contextual_grounding"] = bool(anti.get("contextual_grounding", payload["contextual_grounding"]))
        payload["llm_as_judge"] = bool(anti.get("llm_as_judge", payload["llm_as_judge"]))
        payload["judge_model"] = anti.get("judge_model", payload["judge_model"])

    state.pipeline_form_name = payload["name"]
    state.pipeline_form_description = payload["description"]
    state.pipeline_form_data_source = payload["data_source"]
    state.pipeline_form_selected_models = list(payload["selected_models"])
    state.pipeline_form_technical_prompt = payload["technical_prompt"]
    state.pipeline_form_clinical_prompt = payload["clinical_prompt"]
    state.pipeline_form_ensemble = payload["ensemble"]
    state.pipeline_form_ensemble_size = int(payload["ensemble_size"]) if payload["ensemble_size"] else 1
    state.pipeline_form_chain_of_verification = payload["chain_of_verification"]
    state.pipeline_form_contextual_grounding = payload["contextual_grounding"]
    state.pipeline_form_llm_as_judge = payload["llm_as_judge"]
    state.pipeline_form_judge_model = payload["judge_model"]


def ensure_session_state() -> None:
    state = st.session_state

    state.models = state.get("models", [])
    if "model_edit_id" not in state:
        state.model_edit_id = None
    if "show_model_form" not in state:
        state.show_model_form = False
    if "model_form_error" not in state:
        state.model_form_error = ""
    if "model_db_error" not in state:
        state.model_db_error = ""
    if "models_loaded" not in state:
        sync_models_from_db(state)
        state.models_loaded = True
    model_form_keys = [
        "model_form_name",
        "model_form_description",
        "model_form_model",
        "model_form_temperature",
        "model_form_top_p",
        "model_form_top_k",
        "model_form_num_ctx",
        "model_form_max_output",
    ]
    if any(key not in state for key in model_form_keys):
        reset_model_form_fields(state)

    state.prompts = state.get("prompts", [])
    if "prompt_edit_id" not in state:
        state.prompt_edit_id = None
    if "show_prompt_form" not in state:
        state.show_prompt_form = False
    if "prompt_form_error" not in state:
        state.prompt_form_error = ""
    if "prompt_db_error" not in state:
        state.prompt_db_error = ""
    if "prompts_loaded" not in state:
        sync_prompts_from_db(state)
        state.prompts_loaded = True
    prompt_form_keys = [
        "prompt_form_name",
        "prompt_form_description",
        "prompt_form_type",
        "prompt_form_content",
    ]
    if any(key not in state for key in prompt_form_keys):
        reset_prompt_form_fields(state)

    if "pipelines" not in state:
        state.pipelines = []
    if "pipeline_id_counter" not in state:
        state.pipeline_id_counter = 1
    if "pipeline_edit_id" not in state:
        state.pipeline_edit_id = None
    if "show_pipeline_form" not in state:
        state.show_pipeline_form = False
    if "pipeline_form_error" not in state:
        state.pipeline_form_error = ""
    if "current_pipeline_id" not in state:
        state.current_pipeline_id = None
    if "pipeline_db_error" not in state:
        state.pipeline_db_error = ""
    if "pipelines_loaded" not in state:
        sync_pipelines_from_db(state)
        state.pipelines_loaded = True
    if "pipeline_form_name" not in state:
        reset_pipeline_form_fields(state)


def sanitize_pipeline_state() -> None:
    state = st.session_state
    valid_model_ids = {model["id"] for model in state.models}
    valid_prompt_ids = {prompt["id"] for prompt in state.prompts}

    state.pipeline_form_selected_models = [
        mid for mid in state.pipeline_form_selected_models if mid in valid_model_ids
    ]
    if state.pipeline_form_judge_model not in valid_model_ids:
        state.pipeline_form_judge_model = None
    if state.pipeline_form_technical_prompt not in valid_prompt_ids:
        state.pipeline_form_technical_prompt = None
    if state.pipeline_form_clinical_prompt not in valid_prompt_ids:
        state.pipeline_form_clinical_prompt = None
    if not state.pipeline_form_llm_as_judge:
        state.pipeline_form_judge_model = None

    updated_pipelines = []
    for pipeline in state.pipelines:
        pipeline_copy = deepcopy(pipeline)
        pipeline_copy["selected_models"] = [
            mid for mid in pipeline_copy.get("selected_models", []) if mid in valid_model_ids
        ]
        pipeline_copy["technical_prompt"] = (
            pipeline_copy.get("technical_prompt") if pipeline_copy.get("technical_prompt") in valid_prompt_ids else None
        )
        pipeline_copy["clinical_prompt"] = (
            pipeline_copy.get("clinical_prompt") if pipeline_copy.get("clinical_prompt") in valid_prompt_ids else None
        )
        anti = pipeline_copy.get("anti_hallucination", {})
        judge_model = anti.get("judge_model")
        if judge_model not in valid_model_ids:
            anti["judge_model"] = None
            anti["judgeModel"] = None
        else:
            anti["judgeModel"] = judge_model
        pipeline_copy["anti_hallucination"] = anti
        updated_pipelines.append(pipeline_copy)
    state.pipelines = updated_pipelines

    if state.current_pipeline_id is not None:
        if not any(p["id"] == state.current_pipeline_id for p in state.pipelines):
            state.current_pipeline_id = None


def get_model_label(model_id: int) -> str:
    for model in st.session_state.models:
        if model["id"] == model_id:
            identifier = model.get("model") or "n/a"
            return f"{model.get('name', 'Unnamed')} ({identifier})"
    return "Unknown model"


def get_prompt_label(prompt_id: int | None) -> str:
    if prompt_id is None:
        return "None"
    for prompt in st.session_state.prompts:
        if prompt["id"] == prompt_id:
            return prompt["name"]
    return "Unknown prompt"


def start_new_model() -> None:
    state = st.session_state
    state.model_edit_id = None
    reset_model_form_fields(state)
    state.model_form_error = ""
    state.show_model_form = True


def edit_model(model_id: int) -> None:
    state = st.session_state
    model = next((m for m in state.models if m["id"] == model_id), None)
    if model:
        state.model_edit_id = model_id
        reset_model_form_fields(state, model)
        state.model_form_error = ""
        state.show_model_form = True


def delete_model(model_id: int) -> None:
    state = st.session_state
    try:
        delete_model_from_db(model_id)
        state.model_form_error = ""
    except sqlite3.Error as exc:
        state.model_form_error = f"Failed to delete model: {exc}"
        return

    sync_models_from_db(state)
    sync_pipelines_from_db(state)
    sanitize_pipeline_state()
    if state.model_edit_id == model_id:
        cancel_model_form()


def cancel_model_form() -> None:
    state = st.session_state
    state.show_model_form = False
    state.model_edit_id = None
    state.model_form_error = ""
    reset_model_form_fields(state)


def collect_model_form_data() -> dict:
    state = st.session_state
    return {
        "name": state.model_form_name.strip(),
        "description": state.model_form_description.strip(),
        "model": state.model_form_model.strip(),
        "temperature": float(state.model_form_temperature),
        "top_p": float(state.model_form_top_p),
        "top_k": int(state.model_form_top_k),
        "num_ctx": int(state.model_form_num_ctx),
        "max_output": int(state.model_form_max_output),
    }


def save_model_form() -> None:
    state = st.session_state
    data = collect_model_form_data()
    if not data["name"]:
        state.model_form_error = "Model name is required."
        return
    if not data["model"]:
        state.model_form_error = "Model identifier is required."
        return
    state.model_form_error = ""

    try:
        target_id = state.model_edit_id
        model_id = persist_model_to_db(data, target_id)
    except sqlite3.Error as exc:
        state.model_form_error = f"Failed to save model: {exc}"
        return

    data["id"] = model_id
    sync_models_from_db(state)
    sync_pipelines_from_db(state)
    sanitize_pipeline_state()
    cancel_model_form()


def start_new_prompt() -> None:
    state = st.session_state
    state.prompt_edit_id = None
    reset_prompt_form_fields(state)
    state.prompt_form_error = ""
    state.show_prompt_form = True


def edit_prompt(prompt_id: int) -> None:
    state = st.session_state
    prompt = next((p for p in state.prompts if p["id"] == prompt_id), None)
    if prompt:
        state.prompt_edit_id = prompt_id
        reset_prompt_form_fields(state, prompt)
        state.prompt_form_error = ""
        state.show_prompt_form = True


def delete_prompt(prompt_id: int) -> None:
    state = st.session_state
    try:
        delete_prompt_from_db(prompt_id)
        state.prompt_form_error = ""
    except sqlite3.Error as exc:
        state.prompt_form_error = f"Failed to delete prompt: {exc}"
        return

    sync_prompts_from_db(state)
    sync_pipelines_from_db(state)
    sanitize_pipeline_state()
    if state.prompt_edit_id == prompt_id:
        cancel_prompt_form()


def cancel_prompt_form() -> None:
    state = st.session_state
    state.show_prompt_form = False
    state.prompt_edit_id = None
    state.prompt_form_error = ""
    reset_prompt_form_fields(state)


def collect_prompt_form_data() -> dict:
    state = st.session_state
    return {
        "name": state.prompt_form_name.strip(),
        "description": state.prompt_form_description.strip(),
        "type": state.prompt_form_type,
        "content": state.prompt_form_content.strip(),
    }


def save_prompt_form() -> None:
    state = st.session_state
    data = collect_prompt_form_data()
    if not data["name"]:
        state.prompt_form_error = "Prompt name is required."
        return
    if not data["content"]:
        state.prompt_form_error = "Prompt content cannot be empty."
        return
    state.prompt_form_error = ""

    try:
        target_id = state.prompt_edit_id
        prompt_id = persist_prompt_to_db(data, target_id)
    except sqlite3.Error as exc:
        state.prompt_form_error = f"Failed to save prompt: {exc}"
        return

    data["id"] = prompt_id
    sync_prompts_from_db(state)
    sync_pipelines_from_db(state)
    sanitize_pipeline_state()
    cancel_prompt_form()


def reset_pipeline_form() -> None:
    state = st.session_state
    state.pipeline_edit_id = None
    state.pipeline_form_error = ""
    state.show_pipeline_form = False
    state.current_pipeline_id = None
    reset_pipeline_form_fields(state)


def start_new_pipeline() -> None:
    reset_pipeline_form()
    st.session_state.show_pipeline_form = True


def edit_pipeline(pipeline_id: int) -> None:
    state = st.session_state
    pipeline = next((p for p in state.pipelines if p["id"] == pipeline_id), None)
    if pipeline:
        state.pipeline_edit_id = pipeline_id
        state.current_pipeline_id = pipeline_id
        state.pipeline_form_error = ""
        reset_pipeline_form_fields(state, pipeline)
        state.show_pipeline_form = True


def load_pipeline(pipeline_id: int) -> None:
    state = st.session_state
    pipeline = next((p for p in state.pipelines if p["id"] == pipeline_id), None)
    if pipeline:
        state.pipeline_edit_id = None
        state.pipeline_form_error = ""
        state.current_pipeline_id = pipeline_id
        reset_pipeline_form_fields(state, pipeline)
        state.show_pipeline_form = False


def delete_pipeline(pipeline_id: int) -> None:
    state = st.session_state
    try:
        delete_pipeline_from_db(pipeline_id)
        state.pipeline_form_error = ""
    except sqlite3.Error as exc:
        state.pipeline_form_error = f"Failed to delete pipeline: {exc}"
        return

    sync_pipelines_from_db(state)
    sanitize_pipeline_state()
    if state.current_pipeline_id == pipeline_id:
        state.current_pipeline_id = None
    if state.pipeline_edit_id == pipeline_id:
        reset_pipeline_form()


def collect_pipeline_form_data(include_id: bool = True) -> dict:
    state = st.session_state
    ensemble_flag = bool(state.pipeline_form_ensemble)
    ensemble_size = int(state.pipeline_form_ensemble_size)
    if ensemble_flag:
        ensemble_size = max(ensemble_size, 1)
    chain_flag = bool(state.pipeline_form_chain_of_verification)
    grounding_flag = bool(state.pipeline_form_contextual_grounding)
    judge_flag = bool(state.pipeline_form_llm_as_judge)
    judge_model_id = (
        int(state.pipeline_form_judge_model)
        if judge_flag and state.pipeline_form_judge_model is not None
        else None
    )
    pipeline = {
        "name": state.pipeline_form_name.strip(),
        "description": state.pipeline_form_description.strip(),
        "data_source": state.pipeline_form_data_source.strip(),
        "selected_models": [int(mid) for mid in state.pipeline_form_selected_models],
        "technical_prompt": (
            int(state.pipeline_form_technical_prompt)
            if state.pipeline_form_technical_prompt is not None
            else None
        ),
        "clinical_prompt": (
            int(state.pipeline_form_clinical_prompt)
            if state.pipeline_form_clinical_prompt is not None
            else None
        ),
        "anti_hallucination": {
            "ensemble": ensemble_flag,
            "ensemble_size": ensemble_size,
            "ensembleSize": ensemble_size,
            "chain_of_verification": chain_flag,
            "chainOfVerification": chain_flag,
            "contextual_grounding": grounding_flag,
            "contextualGrounding": grounding_flag,
            "llm_as_judge": judge_flag,
            "llmAsJudge": judge_flag,
            "judge_model": judge_model_id,
            "judgeModel": judge_model_id,
        },
    }
    if include_id and st.session_state.pipeline_edit_id is not None:
        pipeline["id"] = st.session_state.pipeline_edit_id
    return pipeline


def map_pipeline_keys(pipeline: dict) -> dict:
    """Aligns saved pipeline structure with Remix naming for summary rendering."""
    anti = pipeline.get("anti_hallucination", {})
    return {
        "id": pipeline.get("id"),
        "name": pipeline.get("name", ""),
        "description": pipeline.get("description", ""),
        "data_source": pipeline.get("data_source", ""),
        "selected_models": pipeline.get("selected_models", []),
        "technical_prompt": pipeline.get("technical_prompt"),
        "clinical_prompt": pipeline.get("clinical_prompt"),
        "anti_hallucination": {
            "ensemble": bool(anti.get("ensemble", anti.get("ensembleSize", False))),
            "ensemble_size": anti.get("ensemble_size", anti.get("ensembleSize", PIPELINE_DEFAULTS["ensemble_size"])),
            "chain_of_verification": bool(
                anti.get("chain_of_verification", anti.get("chainOfVerification", False))
            ),
            "contextual_grounding": bool(
                anti.get("contextual_grounding", anti.get("contextualGrounding", False))
            ),
            "llm_as_judge": bool(anti.get("llm_as_judge", anti.get("llmAsJudge", False))),
            "judge_model": anti.get("judge_model", anti.get("judgeModel")),
        },
    }


def save_pipeline_form() -> None:
    state = st.session_state
    pipeline = collect_pipeline_form_data()
    if not pipeline["name"]:
        state.pipeline_form_error = "Pipeline name is required."
        return
    if not pipeline["selected_models"]:
        state.pipeline_form_error = "Select at least one model."
        return
    state.pipeline_form_error = ""

    try:
        target_id = state.pipeline_edit_id
        pipeline_id = persist_pipeline_to_db(pipeline, target_id)
    except sqlite3.Error as exc:
        state.pipeline_form_error = f"Failed to save pipeline: {exc}"
        return

    pipeline["id"] = pipeline_id
    sync_pipelines_from_db(state)
    sanitize_pipeline_state()
    reset_pipeline_form()


def get_pipeline_preview() -> dict | None:
    state = st.session_state
    if state.current_pipeline_id is not None:
        pipeline = next((p for p in state.pipelines if p["id"] == state.current_pipeline_id), None)
        return map_pipeline_keys(deepcopy(pipeline)) if pipeline else None
    if state.show_pipeline_form and state.pipeline_form_selected_models:
        preview = collect_pipeline_form_data(include_id=False)
        return map_pipeline_keys(preview)
    return None


def render_reports_selection() -> None:
    state = st.session_state
    state.extractor_bool =False
    # Data loading
    #load data
    @st.cache_data
    def load_data():
        data = pl.scan_parquet("synthetic_data/text_documents.parquet")
        data = data.head(5000)
        return data.with_columns(pl.col("patient_id").cast(pl.Int32))

    data = load_data()
    # Sidebar - user choice
    
    id_col = "patient_id"
    user_ids = (data.select(id_col).unique().collect()
                    .sort(id_col)
                    .get_column(id_col)
                    .to_list())
    
    selected_user = st.sidebar.selectbox("Select Patient Id", user_ids)
    #get the different pipelines
    pipeline_names =[pipeline["name"] for pipeline in load_pipelines_from_db()]
    state.selected_pip = st.sidebar.selectbox("Select Pipeline",pipeline_names)
    st.sidebar.slider("Fuzzy Matching Ratio", 0.0, 1.0,0.8, key="ratio_treshold", step=0.05)
    
    # data filtering based on user selection (ID)
    user_data = data.filter(pl.col(id_col) == selected_user)
    
    # Convert the selected patient information using st-aggrid
    user_data_pd = user_data.collect().to_pandas()
    
    #CSS to delete initial padding
    st.markdown("""
        <style>
            .block-container {
                padding-top: 3rem;
                padding-bottom: 0rem;
                padding-left: 2rem;
                padding-right : 2rem;
            }
            h2 {
                margin-top: 0.5rem;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Layout with an empty column to free space between the AgGrid and the patient's document
    grid_data = user_data_pd[['document_type', 'title', 'datetime']]
    col1, spacer, col2 = st.columns([3.5, 0.1, 6])

    with col1:
        #Grid of medical reports info
        grid_response = configure_aggrid(grid_data, rename_columns = {'document_type': "Document Type", 'title': "Title", 'datetime': "Date"})

    with col2:
    
        state.selec_row = grid_response.get('selected_rows', [])
        
        if state.selec_row is not None and len(state.selec_row) > 0:
            # Extract selected row values
            state.selec_row['datetime'] = pd.to_datetime(state.selec_row['datetime'], errors='coerce')
            doc_type =state.selec_row['document_type'].iloc[0]
            title = state.selec_row['title'].iloc[0]
            date = state.selec_row['datetime'].iloc[0]
            
            # Filter original polars dataframe by the selected row values
            user_data = user_data.with_columns([pl.col("datetime").dt.strftime("%Y-%m-%d").alias("datetime_str")])
            state.filt_data = user_data.filter(
                (pl.col('document_type') == doc_type) &
                (pl.col('title') == title)&
                (pl.col('datetime') == date)
            )
            # Show all matching texts (there could be multiple)
            collected = state.filt_data.select(['text', 'raw_text']).collect()
            text_clean_list = collected['text'].to_list()
            text_raw_list = collected['raw_text'].to_list()
            text_pairs = list(zip(text_clean_list, text_raw_list))
    
            for idx, (state.text_clean, state.text_brut) in enumerate(text_pairs) :
                if "text_brut_view" not in st.session_state:
                    state.text_brut_view = True 
                
                # robust callback for the button
                def toogle_brut():
                    state.text_brut_view = not state.text_brut_view
                
                bouton_texte = "Texte brut" if state.text_brut_view else "Texte nettoyé"
                st.button(bouton_texte, key = "toogle_long_btn", on_click = toogle_brut)
                
                # Display based on the state
                #Bug that has to be resolved : the state button seems not to work
                if state.text_brut_view:
                    display(state.text_clean, False)
                    
                else:
                    display(state.text_brut, False)            
                    
        else:
            st.warning("Sélectionnez un compte rendu dans le tableau pour afficher le document ici.")

def render_ut_extractions()-> None :
    state = st.session_state
    if (state.selec_row is not None) :
        extract_cols = st.columns([0.45,0.55])
        with extract_cols[0] :
            if st.button("Run Extraction") :
                #Get the pipeline info
                target_pipeline = state.selected_pip  # change to the pipeline you need
                pipelines = load_pipelines_from_db()
                models_by_id = {model["id"]: model for model in load_models_from_db()}
                prompts_by_id = {prompt["id"]: prompt for prompt in load_prompts_from_db()}
                    
                pipeline = next(p for p in pipelines if p["name"] == target_pipeline)
                state.model_config = []
                for mid in pipeline["selected_models"] :
                    info = models_by_id[mid]
                    state.model_config.append (
                    {
                    "model_version" : info["model"],
                    "temperature" : info["temperature"],
                    "top_p" :info["top_p"],
                    "top_k": info["top_k"],
                    "num_ctx" : info["num_ctx"],
                    "max_output" : info["max_output"]
                    }
                                        )
                state.system_prompt = prompts_by_id.get(pipeline["technical_prompt"], {}).get("content")
                state.user_prompt = prompts_by_id.get(pipeline["clinical_prompt"], {}).get("content")
    
                #Extract the concepts using the configured llm
                state.prompt = build_messages(state.filt_data,state.system_prompt, state.user_prompt)
                state.extractor_bool = True
    
                ###Run the extraction
                data_extract = llm_extractor(state.prompt, state.filt_data, 
                                             state.model_config[0]["model_version"], state.model_config[0]["num_ctx"], state.model_config[0]["max_output"],
                                             state.model_config[0]["temperature"], state.model_config[0]["top_p"], state.model_config[0]["top_k"], 
                                            )
                state.data_extract = pl.from_pandas(data_extract)
                # Check the structure: list of dicts is easiest to convert to table
                if isinstance(state.data_extract, pl.DataFrame):
                    #In case of errors
                    if not isinstance(state.data_extract["concept_extracted"][0], pl.Series):
                        # st.markdown(data_extract["concept_extracted"][0]["brut"])
                        st.warning("Format de Sortie pas respecté")
                        
                    #In case of good json files    
                    else :
                        json_str = state.data_extract["concept_extracted"][0]
                        json_obj = json_str.to_list()
                        state.json_str = json_obj
                        # print(type(json_str))
                        # now build the Polars DataFrame
                        result = pl.DataFrame(json_str.to_frame("data").unnest("data"))
                        # Display the table
                        st.dataframe(result, use_container_width=True)
                        # st.dataframe(data_extract["positions"][0])
                else :
                    st.warning("sortie dans le mauvais format")

            else:
                st.warning("Utilisez le bouton pour afficher les résulats de l'extraction ici.")

            with extract_cols[1]:
                if "json_str" in state and state.extractor_bool:
                    display(highlight_contexts(state.text_brut, state.json_str, state.ratio_treshold),False)# should be state.extractor_bool but because of the error on our diplay function we will use false
                # else :
                #     st.warning("Aucune tentative d'extraction")
            
    else :
        st.warning("Sélectionner une ligne avant de lancer l'extraction")

def render_raw_output()-> None:
    state = st.session_state
    if state.extractor_bool:
        st.code(state.data_extract["brut_response"][0], language="markdown")
    else :
        st.warning("Aucune tentative d'extraction")
    
def get_available_models() :
    response = requests.get("http://lx181.intra.chu-rennes.fr:11434/api/tags")
    data = response.json()
    model_names = [model["name"] for model in data.get("models", [])]
    return model_names
    
def render_models_tab() -> None:
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("Model Configuration")
        st.write("Define the LLM endpoints that will be available to your pipelines.")
    with header_cols[1]:
        st.button("New model", on_click=start_new_model, type="primary", use_container_width=True)

    if st.session_state.model_db_error:
        st.error(st.session_state.model_db_error)

    if st.session_state.show_model_form:
        header = "Edit model" if st.session_state.model_edit_id is not None else "New model"
        with st.container():
            st.markdown(f"### {header}")
            form_cols = st.columns(2)
            with form_cols[0]:
                st.text_input("Name", key="model_form_name")
            with form_cols[1]:
                st.session_state.model_names = get_available_models()
                st.selectbox(
                    "Model identifier",
                    st.session_state.model_names,
                    key="model_form_model",
                    index = None,
                    placeholder = "Select a model",
                    help="Exact Ollama model tag, e.g. `mistral:7b`.",
                )

            st.text_area(
                "Description",
                key="model_form_description",
                help="Short reminder about latency, pricing, or intended usage.",
            )

            param_cols = st.columns(3)
            with param_cols[0]:
                st.slider("Temperature", 0.0, 2.0, key="model_form_temperature", step=0.05)
                st.slider("Top P", 0.0, 1.0, key="model_form_top_p", step=0.01)
            with param_cols[1]:
                st.number_input("Top K", min_value=1, key="model_form_top_k", step=1)
                st.number_input("Context window (num_ctx)", min_value=1, key="model_form_num_ctx", step=128)
            with param_cols[2]:
                st.number_input("Max output tokens", min_value=1, key="model_form_max_output", step=16)

            if st.session_state.model_form_error:
                st.error(st.session_state.model_form_error)

            col1, col2 = st.columns(2)
            col1.button("Save model", on_click=save_model_form, key="model_form_save", type="primary", use_container_width=True)
            col2.button("Cancel", on_click=cancel_model_form, key="model_form_cancel", use_container_width=True)

    if st.session_state.models:
        st.markdown("### Saved models")
        models = st.session_state.models
        for idx, model in enumerate(models):
            with st.container():
                cols = st.columns([6, 1, 1])
                cols[0].markdown(f"**{model['name']}**  \n{model['description'] or 'No description'}")
                identifier = model.get("model") or "n/a"
                temperature = model.get("temperature")
                top_p = model.get("top_p")
                top_k = model.get("top_k")
                num_ctx = model.get("num_ctx")
                max_output = model.get("max_output")
                temperature_display = f"{temperature:.2f}" if isinstance(temperature, (int, float)) else "n/a"
                top_p_display = f"{top_p:.2f}" if isinstance(top_p, (int, float)) else "n/a"
                top_k_display = f"{top_k}" if isinstance(top_k, (int, float)) else "n/a"
                num_ctx_display = f"{num_ctx}" if isinstance(num_ctx, (int, float)) else "n/a"
                max_output_display = f"{max_output}" if isinstance(max_output, (int, float)) else "n/a"
                minicols = st.columns([0.1, 0.5])
                minicols[0].write(
                    f"- Identifier: `{identifier}`\n"
                    f"- Temperature: {temperature_display}\n"
                    f"- Top P: {top_p_display}\n"
                )
                minicols[1].write(
                    f"- Top K: {top_k_display}\n"
                    f"- Context window (num_ctx): {num_ctx_display}\n"
                    f"- Max output tokens: {max_output_display}")

                cols[1].button("Edit", key=f"edit_model_{model['id']}", on_click=edit_model, args=(model["id"],))
                cols[2].button("Delete", key=f"delete_model_{model['id']}", on_click=delete_model, args=(model["id"],))
            if idx < len(models) - 1:
                st.divider()
    else:
        st.info("No models configured yet. Use *New model* to add one.")


def render_prompts_tab() -> None:
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("Prompt Configuration")
        st.write("Store reusable prompt templates for clinical extraction and technical formatting.")
    with header_cols[1]:
        st.button("New prompt", on_click=start_new_prompt, type="primary", use_container_width=True)

    if st.session_state.prompt_db_error:
        st.error(st.session_state.prompt_db_error)

    if st.session_state.show_prompt_form:
        header = "Edit prompt" if st.session_state.prompt_edit_id is not None else "New prompt"
        with st.container():
            st.markdown(f"### {header}")
            form_cols = st.columns(2)
            with form_cols[0]:
                st.text_input("Prompt name", key="prompt_form_name")
            with form_cols[1]:
                st.selectbox("Prompt type", ["clinical", "technical"], key="prompt_form_type")

            st.text_area("Description", key="prompt_form_description")
            st.text_area("Prompt content", key="prompt_form_content", height=220)

            if st.session_state.prompt_form_error:
                st.error(st.session_state.prompt_form_error)

            col1, col2 = st.columns(2)
            col1.button("Save prompt", on_click=save_prompt_form, key="prompt_form_save", type="primary", use_container_width=True)
            col2.button("Cancel", on_click=cancel_prompt_form, key="prompt_form_cancel", use_container_width=True)

    if st.session_state.prompts:
        st.markdown("### Saved prompts")
        prompts = st.session_state.prompts
        for idx, prompt in enumerate(prompts):
            with st.container():
                cols = st.columns([6, 1, 1])
                badge = "Clinical" if prompt["type"] == "clinical" else "Technical"
                cols[0].markdown(f"**{prompt['name']}** · {badge}")
                cols[0].write(prompt["description"] or "No description")
                # Use the main container area for the expander instead of the column
                with st.expander("Click here to display the prompt"):
                    st.code(prompt["content"], language="markdown")
                cols[1].button("Edit", key=f"edit_prompt_{prompt['id']}", on_click=edit_prompt, args=(prompt["id"],))
                cols[2].button("Delete", key=f"delete_prompt_{prompt['id']}", on_click=delete_prompt, args=(prompt["id"],))
            if idx < len(prompts) - 1:
                st.divider()
    else:
        st.info("No prompts configured yet. Use *New prompt* to add one.")


def render_pipeline_form() -> None:
    st.markdown("### Pipeline editor")

    basics_cols = st.columns(2)
    with basics_cols[0]:
        st.text_input("Pipeline name", key="pipeline_form_name")
    with basics_cols[1]:
        st.text_input("Pipeline description", key="pipeline_form_description")

    st.text_input("Data source", key="pipeline_form_data_source", help="Dataset path, API endpoint, or datasource identifier.")

    st.markdown("#### Model selection")
    model_options = [model["id"] for model in st.session_state.models]
    if model_options:
        st.multiselect(
            "Pick the models that run in parallel",
            model_options,
            key="pipeline_form_selected_models",
            format_func=get_model_label,
        )
    else:
        st.info("No models available. Add models in the *Model Configuration* tab.")
        st.session_state.pipeline_form_selected_models = []

    st.markdown("#### Prompt selection")
    clinical_prompts = [prompt["id"] for prompt in st.session_state.prompts if prompt["type"] == "clinical"]
    technical_prompts = [prompt["id"] for prompt in st.session_state.prompts if prompt["type"] == "technical"]

    cols = st.columns(2)
    cols[0].selectbox(
        "Clinical prompt",
        [None] + clinical_prompts,
        key="pipeline_form_clinical_prompt",
        format_func=get_prompt_label,
    )
    cols[1].selectbox(
        "Technical prompt",
        [None] + technical_prompts,
        key="pipeline_form_technical_prompt",
        format_func=get_prompt_label,
    )

    st.markdown("#### Anti-hallucination strategies")
    st.checkbox("Use ensemble methods", key="pipeline_form_ensemble")
    if st.session_state.pipeline_form_ensemble:
        st.slider("Ensemble size", 1, 10, key="pipeline_form_ensemble_size")

    st.checkbox("Enable chain-of-verification", key="pipeline_form_chain_of_verification")
    st.checkbox("Add contextual grounding", key="pipeline_form_contextual_grounding")
    st.checkbox("Add LLM-as-a-judge validation", key="pipeline_form_llm_as_judge")
    if st.session_state.pipeline_form_llm_as_judge:
        judge_options = [model["id"] for model in st.session_state.models]
        st.selectbox(
            "Judge model",
            [None] + judge_options,
            key="pipeline_form_judge_model",
            format_func=get_model_label,
        )
    else:
        st.session_state.pipeline_form_judge_model = None

    if st.session_state.pipeline_form_error:
        st.error(st.session_state.pipeline_form_error)

    cols = st.columns(2)
    cols[0].button("Save pipeline", on_click=save_pipeline_form, key="pipeline_form_save", type="primary", use_container_width=True)
    cols[1].button("Cancel", on_click=reset_pipeline_form, key="pipeline_form_cancel", use_container_width=True)


def render_pipeline_summary(pipeline: dict) -> None:
    st.markdown("### Pipeline overview")
    cols = st.columns(4)
    cols[0].metric("Models", len(pipeline["selected_models"]))
    cols[1].metric("Strategies", sum(1 for flag in [
        pipeline["anti_hallucination"]["ensemble"],
        pipeline["anti_hallucination"]["chain_of_verification"],
        pipeline["anti_hallucination"]["contextual_grounding"],
        pipeline["anti_hallucination"]["llm_as_judge"],
    ] if flag))
    cols[2].metric("Clinical prompt", "Yes" if pipeline["clinical_prompt"] else "No")
    cols[3].metric("Technical prompt", "Yes" if pipeline["technical_prompt"] else "No")

    st.divider()

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.write(f"**Name:** {pipeline['name'] or 'Unnamed pipeline'}")
        st.write(f"**Description:** {pipeline['description'] or 'No description'}")
    with detail_cols[1]:
        st.write(f"**Data source:** {pipeline['data_source'] or 'Not specified'}")

    st.divider()

    st.markdown("#### Prompts")
    prompt_cols = st.columns(2)
    with prompt_cols[0]:
        st.markdown("**Clinical**")
        st.write(get_prompt_label(pipeline["clinical_prompt"]) if pipeline["clinical_prompt"] else "None")
    with prompt_cols[1]:
        st.markdown("**Technical**")
        st.write(get_prompt_label(pipeline["technical_prompt"]) if pipeline["technical_prompt"] else "None")

    st.divider()

    st.markdown("#### Execution setup")
    execution_cols = st.columns(2)
    with execution_cols[0]:
        st.markdown("**Parallel models**")
        if pipeline["selected_models"]:
            for index, model_id in enumerate(pipeline["selected_models"], start=1):
                st.write(f"{index}. {get_model_label(model_id)}")
        else:
            st.write("No models selected.")

    with execution_cols[1]:
        st.markdown("**Anti-hallucination**")
        anti = pipeline["anti_hallucination"]

        strategies = []
        if anti["ensemble"]:
            strategies.append(f"- Ensemble enabled with size {anti['ensemble_size']}")
        if anti["chain_of_verification"]:
            strategies.append("- Chain-of-verification checks enabled")
        if anti["contextual_grounding"]:
            strategies.append("- Contextual grounding active")
        if anti["llm_as_judge"]:
            judge = get_model_label(anti["judge_model"]) if anti["judge_model"] else "No judge selected"
            strategies.append(f"- LLM-as-a-judge with {judge}")
        if not strategies:
            strategies.append("No additional strategies selected.")

        st.markdown("\n".join(strategies))

    with execution_cols[1]:
        st.markdown("**Anti-hallucination**")
        anti = pipeline["anti_hallucination"]
        if anti["ensemble"]:
            st.write(f"- Ensemble enabled with size {anti['ensemble_size']}")
        if anti["chain_of_verification"]:
            st.write("- Chain-of-verification checks enabled")
        if anti["contextual_grounding"]:
            st.write("- Contextual grounding active")
        if anti["llm_as_judge"]:
            judge = get_model_label(anti["judge_model"]) if anti["judge_model"] else "No judge selected"
            st.write(f"- LLM-as-a-judge with {judge}")
        if not any([
            anti["ensemble"],
            anti["chain_of_verification"],
            anti["contextual_grounding"],
            anti["llm_as_judge"],
        ]):
            st.write("No additional strategies selected.")


def render_pipeline_tab() -> None:
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("Pipeline Management")
        st.write("Combine models, prompts, and anti-hallucination strategies into executable pipelines.")
    with header_cols[1]:
        st.button("New pipeline", on_click=start_new_pipeline, type="primary", use_container_width=True)

    if st.session_state.pipeline_db_error:
        st.error(st.session_state.pipeline_db_error)

    if st.session_state.show_pipeline_form:
        render_pipeline_form()

    if st.session_state.pipelines:
        st.markdown("### Saved pipelines")
        pipelines = st.session_state.pipelines
        for idx, pipeline in enumerate(pipelines):
            with st.container():
                cols = st.columns([5, 1, 1, 1])
                cols[0].markdown(f"**{pipeline['name']}**  \n{pipeline['description'] or 'No description'}")
                cols[0].write(
                    f"- Models: {len(pipeline['selected_models'])}  \n"
                    f"- Strategies: {count_enabled_strategies(pipeline['anti_hallucination'])}"
                )
                cols[1].button("Load", key=f"load_pipeline_{pipeline['id']}", on_click=load_pipeline, args=(pipeline["id"],))
                cols[2].button("Edit", key=f"edit_pipeline_{pipeline['id']}", on_click=edit_pipeline, args=(pipeline["id"],))
                cols[3].button("Delete", key=f"delete_pipeline_{pipeline['id']}", on_click=delete_pipeline, args=(pipeline["id"],))
            if idx < len(pipelines) - 1:
                st.divider()
    else:
        st.info("No pipelines saved yet.")
        
    preview = get_pipeline_preview()
    if preview:
        st.divider()
        render_pipeline_summary(preview)


def render_pipeline_evaluation_tab() -> None:
    st.subheader("Pipeline Evaluation Dashboard")

    state = st.session_state
    defaults = {
        "eval_data_source": "Sample dataset",
        "eval_expert_df": None,
        "eval_llm_df": None,
        "eval_db_path": "medical_eval.db",
        "eval_api_base": "",
        "eval_expert_endpoint": "/expert-annotations",
        "eval_llm_endpoint": "/llm-predictions",
        "eval_api_token": "",
        "eval_remote_format": "json",
    }
    for key, value in defaults.items():
        state.setdefault(key, value)

    data_options = ["Sample dataset", "Upload CSV files", "External tool/DB"]
    default_index = (
        data_options.index(state["eval_data_source"])
        if state["eval_data_source"] in data_options
        else 0
    )
    source = st.selectbox(
        "Dataset source",
        data_options,
        index=default_index,
    )
    if source != state["eval_data_source"]:
        state["eval_expert_df"] = None
        state["eval_llm_df"] = None
        state["eval_data_source"] = source

    db_path_input = st.text_input(
        "Evaluation SQLite path",
        value=state["eval_db_path"],
        help="Used to persist fetched annotations and predictions.",
    )
    if db_path_input:
        state["eval_db_path"] = db_path_input
    db_path = Path(state["eval_db_path"])

    if source == "Sample dataset":
        expert_df, llm_df = load_sample_data()
        state["eval_expert_df"] = expert_df
        state["eval_llm_df"] = llm_df

    elif source == "Upload CSV files":
        upload_cols = st.columns(2)
        with upload_cols[0]:
            uploaded_expert = st.file_uploader(
                "Expert annotations CSV",
                type=["csv"],
                key="eval_expert_upload",
                help="Columns required: note_id, concept, annotator, label.",
            )
        with upload_cols[1]:
            uploaded_llm = st.file_uploader(
                "LLM predictions CSV",
                type=["csv"],
                key="eval_llm_upload",
                help="Columns required: note_id, concept, llm, pipeline, prediction.",
            )

        if uploaded_expert and uploaded_llm:
            expert_df = pl.read_csv(uploaded_expert)
            llm_df = pl.read_csv(uploaded_llm)
            expert_df = expert_df.with_columns(pl.col("label").cast(pl.Int8))
            llm_df = llm_df.with_columns(pl.col("prediction").cast(pl.Int8))
            state["eval_expert_df"] = expert_df
            state["eval_llm_df"] = llm_df
        elif state["eval_expert_df"] is None or state["eval_llm_df"] is None:
            st.info("Upload both CSV files to evaluate pipelines.")
            return

    else:  # External tool/DB
        ensure_database_schema(db_path)
        api_cols = st.columns(2)
        with api_cols[0]:
            api_base = st.text_input(
                "API base URL",
                value=state["eval_api_base"],
                key="eval_api_base_input",
            )
        with api_cols[1]:
            token = st.text_input(
                "Bearer/API token",
                value=state["eval_api_token"],
                key="eval_api_token_input",
                type="password",
            )
        endpoint_cols = st.columns(2)
        with endpoint_cols[0]:
            expert_endpoint = st.text_input(
                "Expert endpoint",
                value=state["eval_expert_endpoint"],
                key="eval_expert_endpoint_input",
            )
        with endpoint_cols[1]:
            llm_endpoint = st.text_input(
                "LLM endpoint",
                value=state["eval_llm_endpoint"],
                key="eval_llm_endpoint_input",
            )
        remote_format = st.selectbox(
            "Remote format",
            options=["json", "csv"],
            index=0 if state["eval_remote_format"] == "json" else 1,
            key="eval_remote_format_select",
        )
        state["eval_api_base"] = api_base
        state["eval_api_token"] = token
        state["eval_expert_endpoint"] = expert_endpoint
        state["eval_llm_endpoint"] = llm_endpoint
        state["eval_remote_format"] = remote_format

        action_cols = st.columns(2)
        fetch_clicked = action_cols[0].button("Fetch remote data", use_container_width=True)
        load_clicked = action_cols[1].button("Load from database", use_container_width=True)

        if fetch_clicked:
            if not api_base or not expert_endpoint or not llm_endpoint:
                st.error("Provide the base URL and both endpoints before fetching.")
            else:
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                with st.spinner("Downloading datasets from the external tool..."):
                    try:
                        expert_remote = fetch_remote_dataset(
                            build_url(api_base, expert_endpoint),
                            expected_format=remote_format,
                            headers=headers,
                        )
                        llm_remote = fetch_remote_dataset(
                            build_url(api_base, llm_endpoint),
                            expected_format=remote_format,
                            headers=headers,
                        )

                        expert_remote = require_columns(
                            expert_remote,
                            ["note_id", "concept", "annotator", "label"],
                            "Expert annotations",
                        ).with_columns(pl.col("label").cast(pl.Int8))
                        llm_remote = require_columns(
                            llm_remote,
                            ["note_id", "concept", "llm", "pipeline", "prediction"],
                            "LLM predictions",
                        ).with_columns(pl.col("prediction").cast(pl.Int8))

                        optional_dtypes = {
                            "prompt_version": pl.Utf8,
                            "retriever": pl.Utf8,
                            "confidence": pl.Float64,
                            "latency_ms": pl.Float64,
                        }
                        for column, dtype in optional_dtypes.items():
                            if column not in llm_remote.columns:
                                llm_remote = llm_remote.with_columns(
                                    pl.lit(None, dtype=dtype).alias(column)
                                )

                        expert_clean = deduplicate_records(
                            expert_remote, subset=["note_id", "concept", "annotator"]
                        )
                        llm_clean = deduplicate_records(
                            llm_remote, subset=["note_id", "concept", "llm", "pipeline"]
                        )

                        append_dataframe_to_db(
                            expert_remote,
                            "expert_annotations",
                            db_path,
                            source_label="external-download",
                        )
                        append_dataframe_to_db(
                            llm_remote,
                            "llm_predictions",
                            db_path,
                            source_label="external-download",
                        )

                        state["eval_expert_df"] = expert_clean
                        state["eval_llm_df"] = llm_clean
                        st.success(
                            f"Fetched {expert_clean.height} expert rows and {llm_clean.height} LLM rows."
                        )
                    except Exception as exc:
                        st.error(f"Failed to download data: {exc}")

        if load_clicked:
            expert_loaded = load_table_from_db("expert_annotations", db_path)
            llm_loaded = load_table_from_db("llm_predictions", db_path)
            if expert_loaded.is_empty() or llm_loaded.is_empty():
                st.warning("No datasets stored in the evaluation database yet.")
            else:
                expert_clean = deduplicate_records(
                    expert_loaded, subset=["note_id", "concept", "annotator"]
                )
                llm_clean = deduplicate_records(
                    llm_loaded, subset=["note_id", "concept", "llm", "pipeline"]
                )
                state["eval_expert_df"] = expert_clean
                state["eval_llm_df"] = llm_clean
                st.success(
                    f"Loaded {expert_clean.height} expert rows and {llm_clean.height} prediction rows from the database."
                )

    expert_df = state.get("eval_expert_df")
    llm_df = state.get("eval_llm_df")
    if expert_df is None or llm_df is None:
        st.info("Provide expert annotations and LLM predictions to view evaluation metrics.")
        return

    if expert_df.is_empty() or llm_df.is_empty():
        st.warning("Loaded datasets are empty. Adjust the source or reload data.")
        return

    concept_options = sorted(expert_df.get_column("concept").unique().to_list())
    if not concept_options:
        st.warning("No concepts found in the expert annotations.")
        return

    llm_options = sorted(llm_df.get_column("llm").unique().to_list())
    pipeline_options = sorted(llm_df.get_column("pipeline").unique().to_list())

    filter_cols = st.columns(3)
    selected_concepts = filter_cols[0].multiselect(
        "Concepts",
        options=concept_options,
        default=concept_options,
        key="eval_filter_concepts",
    )
    selected_llms = filter_cols[1].multiselect(
        "LLMs",
        options=llm_options,
        default=llm_options,
        key="eval_filter_llms",
    )
    selected_pipelines = filter_cols[2].multiselect(
        "Pipelines",
        options=pipeline_options,
        default=pipeline_options,
        key="eval_filter_pipelines",
    )

    concept_filter = selected_concepts or concept_options
    llm_filter = selected_llms or llm_options
    pipeline_filter = selected_pipelines or pipeline_options

    gold = compute_majority_vote(expert_df)
    pairwise = compute_pairwise_kappa(expert_df, concept_filter)
    fleiss = compute_fleiss_kappa_per_concept(expert_df, concept_filter)
    krippendorff = compute_krippendorff_alpha_per_concept(expert_df, concept_filter)
    coverage = compute_gold_coverage(expert_df, gold, concept_filter)
    metrics = prepare_llm_metrics(
        llm_df,
        gold,
        concept_filter,
        llm_filter,
        pipeline_filter,
    )

    best_f1 = None
    if not metrics.empty:
        f1_summary = metrics.groupby("llm")["f1"].mean()
        if not f1_summary.empty:
            best_f1 = f1_summary.max()

    def format_metric(value: float | None, pattern: str) -> str:
        if value is None or pd.isna(value):
            return "n/a"
        return pattern.format(value)

    kappa_mean = pairwise["kappa"].mean() if not pairwise.empty else None
    alpha_mean = (
        krippendorff["krippendorff_alpha"].mean()
        if not krippendorff.empty
        else None
    )

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Mean κ", format_metric(kappa_mean, "{:.2f}"))
    kpi_cols[1].metric("Best LLM F1", format_metric(best_f1, "{:.2%}"))
    kpi_cols[2].metric(
        "Gold coverage",
        format_metric(coverage, "{:.0%}"),
        help="Proportion of note/concept pairs with a resolved gold label.",
    )
    kpi_cols[3].metric("Krippendorff α", format_metric(alpha_mean, "{:.2f}"))

    st.markdown("#### Pairwise κ by concept")
    if pairwise.empty:
        st.info("No overlapping annotations found for the selected concepts.")
    else:
        st.dataframe(
            pairwise.sort_values(["concept", "kappa"], ascending=[True, False])
            .style.format({"kappa": "{:.3f}", "n_samples": "{:d}"}),
            use_container_width=True,
        )

    st.markdown("#### Fleiss κ summary")
    if fleiss.empty:
        st.info("Unable to compute Fleiss κ with the current selection.")
    else:
        st.dataframe(
            fleiss.sort_values("concept")
            .style.format({"fleiss_kappa": "{:.3f}", "n_items": "{:d}"}),
            use_container_width=True,
        )

    st.markdown("#### Krippendorff α (nominal)")
    if krippendorff.empty:
        st.info("Krippendorff α requires at least two annotations per note/concept.")
    else:
        st.dataframe(
            krippendorff.sort_values("concept")
            .style.format(
                {
                    "krippendorff_alpha": "{:.3f}",
                    "n_items": "{:d}",
                    "mean_raters": "{:.1f}",
                }
            ),
            use_container_width=True,
        )

    st.markdown("### LLM performance vs gold standard")
    if metrics.empty:
        st.warning("No LLM predictions align with the current filters.")
        return

    detail = (
        metrics.groupby(["llm", "concept"])
        .agg(
            f1=("f1", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            specificity=("specificity", "mean"),
            support=("support", "sum"),
        )
        .reset_index()
    )
    st.dataframe(
        detail.sort_values(["concept", "f1"], ascending=[True, False])
        .style.format(
            {
                "f1": "{:.2f}",
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "specificity": "{:.2f}",
                "support": "{:d}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("#### Pipeline comparison")
    pipeline_summary = (
        metrics.groupby(["llm", "pipeline"])
        .agg(
            f1=("f1", "mean"),
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            specificity=("specificity", "mean"),
            confidence=("confidence", "mean"),
            latency_ms=("latency_ms", "mean"),
            support=("support", "sum"),
        )
        .reset_index()
    )
    st.dataframe(
        pipeline_summary.sort_values("f1", ascending=False)
        .style.format(
            {
                "f1": "{:.2f}",
                "precision": "{:.2f}",
                "recall": "{:.2f}",
                "specificity": "{:.2f}",
                "confidence": "{:.2f}",
                "latency_ms": "{:.0f}",
                "support": "{:d}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("#### Error distribution")
    error_summary = (
        metrics.groupby("llm")[["tp", "tn", "fp", "fn"]]
        .sum()
        .reset_index()
    )
    st.dataframe(error_summary, use_container_width=True)

    if {"confidence"}.issubset(set(llm_df.columns)):
        filtered_predictions = llm_df.filter(
            pl.col("llm").is_in(llm_filter)
            & pl.col("pipeline").is_in(pipeline_filter)
            & pl.col("concept").is_in(concept_filter)
        )
        if not filtered_predictions.is_empty():
            preds_pd = filtered_predictions.to_pandas()
            if "confidence" in preds_pd.columns:
                preds_pd = preds_pd.dropna(subset=["confidence"])
                if not preds_pd.empty:
                    preds_pd["confidence_bucket"] = pd.cut(
                        preds_pd["confidence"],
                        bins=[0.0, 0.5, 0.7, 0.85, 1.0],
                        labels=["≤0.50", "0.51–0.70", "0.71–0.85", ">0.85"],
                        include_lowest=True,
                    )
                    distribution = (
                        preds_pd.groupby(["confidence_bucket", "llm"])
                        .size()
                        .unstack(fill_value=0)
                    )
                    st.markdown("#### Confidence distribution")
                    st.bar_chart(distribution)


def main() -> None:
    ensure_session_state()
    sanitize_pipeline_state()

    tabs = st.tabs(["Pipeline Configuration", "Pipeline Unit Test", " Pipeline Evaluation"])
    with tabs[0]:
        pip_tabs = st.tabs(["Models", "Prompts", "Pipeline"])
        with pip_tabs[0]:
            render_models_tab()
        with pip_tabs[1]:
            render_prompts_tab()
        with pip_tabs[2]:
            render_pipeline_tab()
    with tabs[1] :
        ut_tabs = st.tabs(["Report Selection","Extraction", "Raw Output"])
        with ut_tabs[0]:
            render_reports_selection()
        with ut_tabs[1]:
            render_ut_extractions()
        with ut_tabs[2]:
            render_raw_output()
    with tabs[2]:
        render_pipeline_evaluation_tab()

        
    

if __name__ == "__main__":
    main()
