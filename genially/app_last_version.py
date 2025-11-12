import sqlite3
from copy import deepcopy
from pathlib import Path
import time
from collections import defaultdict
import polars as pl
import pandas as pd
from datetime import datetime
import json
import requests
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

from initialize_pipeline_db import DB_FILENAME, SCHEMA, ensure_database
from display_text import display, highlight_contexts, configure_aggrid
from llm_extractor_pipeline import build_messages, llm_extractor
from evaluation_pipeline import run_pipeline_prediction, compute_concept_metrics

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
    if "gold_eval_result" not in state:
        state.gold_eval_result = None
    if "gold_eval_error" not in state:
        state.gold_eval_error = ""


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


def hydrate_pipeline_runtime(pipeline: dict) -> dict:
    """Attach prompt/model metadata for execution time."""
    sanitized = map_pipeline_keys(pipeline)
    models_by_id = {model["id"]: model for model in load_models_from_db()}
    prompts_by_id = {prompt["id"]: prompt for prompt in load_prompts_from_db()}

    model_confs = [
        models_by_id[mid]
        for mid in sanitized["selected_models"]
        if mid in models_by_id
    ]

    anti = sanitized["anti_hallucination"]
    anti_config = {
        "ensemble": bool(anti.get("ensemble")),
        "ensemble_size": int(
            anti.get("ensemble_size", anti.get("ensembleSize", PIPELINE_DEFAULTS["ensemble_size"]))
            or PIPELINE_DEFAULTS["ensemble_size"]
        ),
        "chain_of_verification": bool(anti.get("chain_of_verification")),
        "contextual_grounding": bool(anti.get("contextual_grounding")),
        "llm_as_judge": bool(anti.get("llm_as_judge")),
        "judge_model": anti.get("judge_model"),
    }

    system_prompt = prompts_by_id.get(sanitized["technical_prompt"], {}).get("content")
    user_prompt = prompts_by_id.get(sanitized["clinical_prompt"], {}).get("content")
    judge_conf = None
    judge_id = anti_config.get("judge_model")
    if judge_id in models_by_id:
        judge_conf = models_by_id[judge_id]

    return {
        "id": sanitized["id"],
        "name": sanitized["name"],
        "models": model_confs,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "anti": anti_config,
        "judge_model": judge_conf,
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
    response = requests.get("http://localhost:11435/api/tags")
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



##### Evaluation Tab Functions #####

ANNOTATIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    n_cols INTEGER NOT NULL,
    data_blob BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"""


def ensure_annotations_table() -> None:
    """Ensure the annotations table exists alongside the pipeline tables."""
    with get_db_connection() as conn:
        conn.executescript(ANNOTATIONS_SCHEMA)


def dataframe_to_blob(df: pl.DataFrame) -> bytes:
    """Serialize a Polars DataFrame into a Parquet blob for storage."""
    buffer = BytesIO()
    df.write_parquet(buffer)
    return buffer.getvalue()


def blob_to_dataframe(blob: bytes) -> pl.DataFrame:
    """Deserialize a Parquet blob back into a Polars DataFrame."""
    return pl.read_parquet(BytesIO(blob))


def reset_annotation_sequence(conn: sqlite3.Connection) -> None:
    """Align sqlite_sequence with the current max annotation id."""
    max_id = conn.execute("SELECT COALESCE(MAX(id), 0) AS max_id FROM annotations").fetchone()[0]
    if max_id == 0:
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'annotations'")
    else:
        cursor = conn.execute(
            "UPDATE sqlite_sequence SET seq = ? WHERE name = 'annotations'",
            (max_id,),
        )
        if cursor.rowcount == 0:
            conn.execute(
                "INSERT INTO sqlite_sequence(name, seq) VALUES ('annotations', ?)",
                (max_id,),
            )


def load_annotations_from_db() -> list[dict]:
    """Load stored annotations from SQLite."""
    ensure_annotations_table()
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, filename, timestamp, n_rows, n_cols, data_blob
            FROM annotations
            ORDER BY id
            """
        ).fetchall()

    annotations: list[dict] = []
    for row in rows:
        annotations.append(
            {
                "id": row["id"],
                "filename": row["filename"],
                "timestamp": row["timestamp"],
                "n_rows": row["n_rows"],
                "n_cols": row["n_cols"],
                "data": blob_to_dataframe(bytes(row["data_blob"])),
            }
        )
    return annotations


# Initialize session state for database
if "annotations_db" not in st.session_state:
    st.session_state.annotations_db = []

if "annotations_loaded" not in st.session_state:
    st.session_state.annotations_db = load_annotations_from_db()
    st.session_state.annotations_loaded = True


def load_csv_file(uploaded_file):
    """Load CSV file into Polars DataFrame"""
    try:
        df = pl.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {str(e)}")
        return None

def save_annotation_to_db(df, filename):
    """Persist an annotation to SQLite and session state."""
    ensure_annotations_table()
    timestamp = datetime.now().isoformat()
    payload = (
        filename,
        timestamp,
        df.height,
        df.width,
        sqlite3.Binary(dataframe_to_blob(df)),
    )

    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO annotations (filename, timestamp, n_rows, n_cols, data_blob)
            VALUES (?, ?, ?, ?, ?)
            """,
            payload,
        )
        annotation_id = cursor.lastrowid

    annotation_entry = {
        "id": annotation_id,
        "filename": filename,
        "timestamp": timestamp,
        "data": df,
        "n_rows": df.height,
        "n_cols": df.width,
    }
    st.session_state.annotations_db.append(annotation_entry)
    return annotation_id

def delete_annotation(annotation_id):
    """Delete an annotation, resequence IDs, and refresh session cache."""
    ensure_annotations_table()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        conn.execute("UPDATE annotations SET id = id - 1 WHERE id > ?", (annotation_id,))
        reset_annotation_sequence(conn)

    st.session_state.annotations_db = load_annotations_from_db()


def clear_all_annotations():
    """Remove every annotation from the persistent store and session."""
    ensure_annotations_table()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM annotations")
        reset_annotation_sequence(conn)
    st.session_state.annotations_db = []

def compute_cohen_kappa_per_concept(df1, df2, id_col, concept_cols):
    """
    Compute Cohen's Kappa between two annotators for each medical concept.
    Each concept column contains binary values (0 or 1).
    Returns dict of {concept: kappa_score}
    """
    # Merge on ID to align reports
    merged = df1.join(
        df2, 
        on=id_col, 
        how='inner',
        suffix='_ann2'
    )
    
    if merged.height == 0:
        return {}, 0
    
    kappa_scores = {}
    n = merged.height
    
    for concept in concept_cols:
        col1 = concept
        col2 = f"{concept}_ann2"
        
        if col1 not in merged.columns or col2 not in merged.columns:
            continue
        
        # Calculate observed agreement
        agreements = (merged[col1] == merged[col2]).sum()
        observed_agreement = agreements / n
        
        # Calculate expected agreement for binary classification
        # P(both say 1) + P(both say 0)
        p1_1 = merged[col1].sum() / n  # annotator 1 says 1
        p2_1 = merged[col2].sum() / n  # annotator 2 says 1
        p1_0 = 1 - p1_1  # annotator 1 says 0
        p2_0 = 1 - p2_1  # annotator 2 says 0
        
        expected_agreement = (p1_1 * p2_1) + (p1_0 * p2_0)
        
        # Cohen's Kappa
        if expected_agreement == 1:
            kappa = 1.0
        else:
            kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        
        kappa_scores[concept] = kappa
    
    return kappa_scores, n

def compute_fleiss_kappa_per_concept(dfs, id_col, concept_cols):
    """
    Compute Fleiss' Kappa for multiple annotators for each medical concept.
    Returns dict of {concept: kappa_score}
    """
    if len(dfs) < 2:
        return {}
    
    kappa_scores = {}
    
    for concept in concept_cols:
        # Get all IDs that appear in all dataframes
        common_ids = set(dfs[0][id_col].unique().to_list())
        for df in dfs[1:]:
            common_ids = common_ids.intersection(set(df[id_col].unique().to_list()))
        
        if len(common_ids) == 0:
            continue
        
        common_ids = sorted(list(common_ids))
        
        # Build rating matrix
        rating_matrix = []
        n_annotators = len(dfs)
        
        for report_id in common_ids:
            # Get annotations from all annotators for this report
            annotations = []
            for df in dfs:
                report_data = df.filter(pl.col(id_col) == report_id)
                if report_data.height > 0 and concept in report_data.columns:
                    annotations.append(int(report_data[concept][0]))
                else:
                    annotations.append(None)
            
            # Only include if all annotators have annotated
            if None not in annotations:
                # Count 0s and 1s
                count_0 = annotations.count(0)
                count_1 = annotations.count(1)
                rating_matrix.append([count_0, count_1])
        
        if len(rating_matrix) == 0:
            continue
        
        N = len(rating_matrix)  # number of subjects (reports)
        n = n_annotators
        
        # Calculate P_i
        P_values = []
        for row in rating_matrix:
            sum_sq = sum(x**2 for x in row)
            P_i = (sum_sq - n) / (n * (n - 1))
            P_values.append(P_i)
        
        P_bar = sum(P_values) / N
        
        # Calculate P_e
        p_0 = sum(rating_matrix[i][0] for i in range(N)) / (N * n)
        p_1 = sum(rating_matrix[i][1] for i in range(N)) / (N * n)
        
        P_e = p_0**2 + p_1**2
        
        # Fleiss' Kappa
        if P_e == 1:
            kappa = 1.0
        else:
            kappa = (P_bar - P_e) / (1 - P_e)
        
        kappa_scores[concept] = kappa
    
    return kappa_scores

def plot_pairwise_agreement(annotations_db, id_col, concept_cols):
    """Create heatmap of average pairwise Cohen's Kappa scores across all concepts"""
    n = len(annotations_db)
    if n < 2:
        return None
    
    avg_kappa_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    count_matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                avg_kappa_matrix[i][j] = 1.0
                count_matrix[i][j] = annotations_db[i]['data'].height
            elif i < j:
                kappa_dict, count = compute_cohen_kappa_per_concept(
                    annotations_db[i]['data'],
                    annotations_db[j]['data'],
                    id_col,
                    concept_cols
                )
                if kappa_dict:
                    avg_kappa = sum(kappa_dict.values()) / len(kappa_dict)
                    avg_kappa_matrix[i][j] = avg_kappa
                    avg_kappa_matrix[j][i] = avg_kappa
                    count_matrix[i][j] = count
                    count_matrix[j][i] = count
    
    labels = [f"Ann{i+1}: {ann['filename'][:20]}" for i, ann in enumerate(annotations_db)]
    
    fig = go.Figure(data=go.Heatmap(
        z=avg_kappa_matrix,
        x=labels,
        y=labels,
        colorscale='RdYlGn',
        zmid=0.5,
        text=[[f"κ={avg_kappa_matrix[i][j]:.3f}<br>n={count_matrix[i][j]}" 
               for j in range(n)] for i in range(n)],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Avg Cohen's Kappa")
    ))
    
    fig.update_layout(
        title="Pairwise Inter-Annotator Agreement (Average Cohen's Kappa)",
        xaxis_title="Annotator",
        yaxis_title="Annotator",
        height=600
    )
    
    return fig


def compute_gold_standard_dataset(
    annotations: list[dict],
    id_col: str | None,
    concept_cols: list[str],
    text_col: str | None = None,
) -> pl.DataFrame:
    """Build a consensus dataset via majority voting across annotators."""
    if not annotations or not id_col or not concept_cols:
        return pl.DataFrame()

    vote_cache: dict = defaultdict(lambda: defaultdict(list))
    text_cache: dict = {}

    for ann in annotations:
        df = ann["data"]
        if id_col not in df.columns:
            continue

        present_concepts = [col for col in concept_cols if col in df.columns]
        if not present_concepts:
            continue

        select_cols = [id_col] + present_concepts
        include_text = text_col and text_col in df.columns
        if include_text:
            select_cols.append(text_col)

        rows = df.select(select_cols).to_dicts()
        for row in rows:
            record_id = row.get(id_col)
            if record_id is None:
                continue

            if include_text and record_id not in text_cache:
                text_value = row.get(text_col)
                if text_value is not None:
                    text_cache[record_id] = text_value

            for concept in present_concepts:
                value = row.get(concept)
                if value is None:
                    continue
                try:
                    vote_cache[record_id][concept].append(int(value))
                except (TypeError, ValueError):
                    continue

    records: list[dict] = []
    for record_id, concept_votes in vote_cache.items():
        entry = {id_col: record_id}
        if text_col:
            entry[text_col] = text_cache.get(record_id)
        has_label = False

        for concept in concept_cols:
            votes = concept_votes.get(concept, [])
            if votes:
                positives = sum(1 for vote in votes if vote == 1)
                negatives = len(votes) - positives
                entry[concept] = 1 if positives >= negatives else 0
                has_label = True
            else:
                entry[concept] = None

        if has_label:
            records.append(entry)

    return pl.DataFrame(records) if records else pl.DataFrame()


def derive_annotation_columns(annotations: list[dict]) -> tuple[str | None, list[str]]:
    """Infer the ID column and concept columns from the first annotation table."""
    if not annotations:
        return None, []

    sample_df = annotations[0]["data"]
    columns = sample_df.columns
    id_col = next((col for col in columns if "id" in col.lower()), None)

    excluded = {"id", "report_text", "text", "report"}
    concept_cols = [
        col
        for col in columns
        if col.lower() not in excluded and col != id_col
    ]
    return id_col, concept_cols


def detect_report_text_column(annotations: list[dict]) -> str | None:
    """Identify a report text column if one exists."""
    if not annotations:
        return None

    columns = annotations[0]["data"].columns
    preferred = ["report_text", "text", "report"]
    for candidate in preferred:
        if candidate in columns:
            return candidate

    for column in columns:
        if "text" in column.lower():
            return column

    return None


def render_annotations_sidebar(annotations: list[dict]) -> tuple[str | None, list[str]]:
    """Show database summary metrics and return inferred column metadata."""
    id_col, concept_cols = derive_annotation_columns(annotations)

    with st.sidebar:
        st.subheader("Database Summary")
        st.metric("Total Annotations", len(annotations))
        if annotations:
            total_reports = sum(ann["n_rows"] for ann in annotations)
            st.metric("Total Reports", total_reports)
            st.metric("Medical Concepts", len(concept_cols))
        else:
            st.info("Upload annotation CSV files to populate the evaluation database.")

    return id_col, concept_cols


def render_annotations_management_tab() -> None:
    """Uploader, preview, and CRUD controls for annotation files."""
    uploaded_files = st.file_uploader(
        "Upload CSV annotation files",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files containing medical concept annotations",
    )
    st.divider()

    if uploaded_files:
        st.markdown("##### Preview Uploaded Files")
        for uploaded_file in uploaded_files:
            with st.expander(f"File: {uploaded_file.name}"):
                df = load_csv_file(uploaded_file)
                if df is None:
                    continue

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.dataframe(df.head(10), use_container_width=True)
                with col2:
                    st.metric("Rows", df.height)
                    st.metric("Columns", df.width)
                with col3:
                    if st.button(
                        "Save to Database", key=f"save_{uploaded_file.name}"
                    ):
                        save_annotation_to_db(df, uploaded_file.name)
                        st.success(f"Saved {uploaded_file.name}")
                        st.rerun()

    st.markdown("##### Preview Annotations Database")
    annotations = st.session_state.annotations_db

    if not annotations:
        st.info("No annotations saved yet. Upload files above to build the evaluation set.")
        return

    for ann in annotations:
        with st.expander(f"{ann['filename']} (ID: {ann['id']})"):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**Uploaded:** {ann['timestamp']}")
                st.write(f"**Rows:** {ann['n_rows']} | **Columns:** {ann['n_cols']}")

            with col2:
                csv_buffer = BytesIO()
                ann["data"].write_csv(csv_buffer)
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=ann["filename"],
                    mime="text/csv",
                    key=f"download_{ann['id']}",
                )

            with col3:
                if st.button("Delete", key=f"delete_{ann['id']}", type="secondary"):
                    delete_annotation(ann["id"])
                    st.success(f"Deleted {ann['filename']}")
                    st.rerun()

            st.dataframe(ann["data"].head(20), use_container_width=True)

    if st.button("Clear All Annotations", type="primary"):
        clear_all_annotations()
        st.success("All annotations cleared!")
        st.rerun()


def render_inter_annotator_analysis_tab(
    id_col: str | None, concept_cols: list[str]
) -> None:
    """Visualizations and agreement metrics for saved annotations."""
    st.header("Inter-Annotator Agreement Analysis")
    annotations = st.session_state.annotations_db

    if len(annotations) < 2:
        st.warning("Upload and save at least two annotation files to compare agreement.")
        return

    if not id_col:
        st.error("Could not auto-detect an ID column. Ensure one column contains 'id'.")
        return

    sample_df = annotations[0]["data"]
    missing_concepts = [col for col in concept_cols if col not in sample_df.columns]
    valid_concepts = [col for col in concept_cols if col in sample_df.columns]

    if missing_concepts:
        st.warning(f"Some concept columns were not found: {', '.join(missing_concepts)}")

    if not valid_concepts:
        st.error("No valid concept columns detected. Update the source data or column rules.")
        return

    st.subheader("Overall Agreement Metrics")

    dfs = [ann["data"] for ann in annotations]
    fleiss_scores = compute_fleiss_kappa_per_concept(dfs, id_col, valid_concepts)

    if fleiss_scores:
        avg_fleiss = sum(fleiss_scores.values()) / len(fleiss_scores)
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Average Fleiss' Kappa",
                f"{avg_fleiss:.3f}",
                help="Average agreement across all concepts and annotators",
            )

            if avg_fleiss < 0:
                interpretation = "Poor (Less than chance)"
            elif avg_fleiss < 0.20:
                interpretation = "Slight"
            elif avg_fleiss < 0.40:
                interpretation = "Fair"
            elif avg_fleiss < 0.60:
                interpretation = "Moderate"
            elif avg_fleiss < 0.80:
                interpretation = "Substantial"
            else:
                interpretation = "Almost Perfect"

            st.info(f"Interpretation: **{interpretation}**")

        with col2:
            n = len(annotations)
            all_kappa_scores: list[float] = []
            for i in range(n):
                for j in range(i + 1, n):
                    kappa_dict, _ = compute_cohen_kappa_per_concept(
                        annotations[i]["data"],
                        annotations[j]["data"],
                        id_col,
                        valid_concepts,
                    )
                    if kappa_dict:
                        all_kappa_scores.extend(kappa_dict.values())

            if all_kappa_scores:
                avg_pairwise = sum(all_kappa_scores) / len(all_kappa_scores)
                st.metric(
                    "Average Pairwise Cohen's Kappa",
                    f"{avg_pairwise:.3f}",
                    help="Mean of all pairwise agreements across all concepts",
                )
                st.metric("Number of Comparisons", len(all_kappa_scores))

        st.divider()
        st.subheader("Agreement by Medical Concept (Fleiss' Kappa)")

        concept_df = pl.DataFrame(
            [
                {"Medical Concept": concept, "Fleiss' Kappa": round(score, 3)}
                for concept, score in fleiss_scores.items()
            ]
        )
        st.dataframe(concept_df, use_container_width=True, hide_index=True)

        fig_fleiss = px.bar(
            concept_df.to_pandas(),
            x="Medical Concept",
            y="Fleiss' Kappa",
            title="Agreement per Concept",
        )
        fig_fleiss.add_hline(y=0.6, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_fleiss, use_container_width=True)

    st.subheader("Pairwise Agreement Matrix")
    fig = plot_pairwise_agreement(annotations, id_col, valid_concepts)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Detailed Pairwise Agreement (Cohen's Kappa)")

    ann1_idx = st.selectbox(
        "Select Annotator 1",
        range(len(annotations)),
        format_func=lambda i: annotations[i]["filename"],
    )
    ann2_idx = st.selectbox(
        "Select Annotator 2",
        range(len(annotations)),
        format_func=lambda i: annotations[i]["filename"],
        index=1 if len(annotations) > 1 else 0,
    )

    if ann1_idx == ann2_idx:
        st.info("Select two different annotators to compare detailed pairwise metrics.")
    else:
        kappa_dict, n_common = compute_cohen_kappa_per_concept(
            annotations[ann1_idx]["data"],
            annotations[ann2_idx]["data"],
            id_col,
            valid_concepts,
        )

        if kappa_dict:
            st.info(f"Comparing {n_common} common reports.")
            pairwise_scores = [
                {"Medical Concept": concept, "Cohen's Kappa": round(kappa, 3)}
                for concept, kappa in sorted(
                    kappa_dict.items(), key=lambda item: item[1], reverse=True
                )
            ]
            pairwise_df = pl.DataFrame(pairwise_scores)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(pairwise_df, use_container_width=True, hide_index=True)
            with col2:
                fig_pair = px.bar(
                    pairwise_df.to_pandas(),
                    x="Medical Concept",
                    y="Cohen's Kappa",
                    title="Agreement per Concept",
                )
                fig_pair.add_hline(y=0.6, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_pair, use_container_width=True)
        else:
            st.warning("No common reports were found between the selected annotators.")

    st.divider()
    st.subheader("Annotation Distribution by Concept")

    dist_data = []
    for ann in annotations:
        for concept in valid_concepts:
            if concept in ann["data"].columns:
                positive = ann["data"][concept].sum()
                total = ann["data"].height
                dist_data.append(
                    {
                        "Annotator": ann["filename"],
                        "Concept": concept,
                        "Positive (1)": positive,
                        "Negative (0)": total - positive,
                        "Prevalence %": round((positive / total * 100), 1)
                        if total > 0
                        else 0,
                    }
                )

    if dist_data:
        dist_df = pl.DataFrame(dist_data)
        fig_dist = go.Figure()
        for annotator in dist_df["Annotator"].unique():
            annotator_data = dist_df.filter(pl.col("Annotator") == annotator)
            fig_dist.add_trace(
                go.Bar(
                    name=annotator,
                    x=annotator_data["Concept"].to_list(),
                    y=annotator_data["Prevalence %"].to_list(),
                    text=annotator_data["Prevalence %"].to_list(),
                    texttemplate="%{text}%",
                    textposition="auto",
                )
            )

        fig_dist.update_layout(
            title="Positive Annotation Rate by Concept and Annotator",
            xaxis_title="Medical Concept",
            yaxis_title="Prevalence (%)",
            barmode="group",
            height=500,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        with st.expander("Detailed Distribution Table"):
            st.dataframe(dist_df, use_container_width=True, hide_index=True)


def render_gold_standard_tab(id_col: str | None, concept_cols: list[str]) -> None:
    """Display consensus dataset and provide a mock evaluation workflow."""
   
    annotations = st.session_state.annotations_db

    if not annotations:
        st.info("Upload annotations to build the gold standard dataset.")
        return

    if not id_col or not concept_cols:
        st.warning("Unable to infer ID or concept columns from the annotations.")
        return

    text_col = detect_report_text_column(annotations)
    gold_df = compute_gold_standard_dataset(
        annotations, id_col, concept_cols, text_col=text_col
    )
    if gold_df.is_empty():
        st.warning("Could not compute a gold standard dataset from the current annotations.")
        return
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
    col2.metric("Gold Standard Records", gold_df.height)
    col3.metric("Concept Columns", len(concept_cols))

    if text_col:
        st.caption(f"Including report text column: `{text_col}`")

    with st.expander("View Gold Standard Dataset"):
        st.dataframe(gold_df, use_container_width=True, hide_index=True)

    csv_buffer = BytesIO()
    gold_df.write_csv(csv_buffer)
    st.download_button(
        label="Download Gold Standard CSV",
        data=csv_buffer.getvalue(),
        file_name="gold_standard.csv",
        mime="text/csv",
    )

    train_pct = st.slider("Training Split (%)", min_value=10, max_value=95, value=80, step=5)
    total_rows = gold_df.height
    train_rows = max(0, int(total_rows * train_pct / 100))
    test_rows = total_rows - train_rows

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Records", train_rows)
    with col2:
        st.metric("Testing Records", test_rows)

    pipelines = load_pipelines_from_db()
    selected_pipeline_id = None
    pipeline_name_lookup: dict[int, str] = {}

    if pipelines:
        pipeline_name_lookup = {
            pipeline["id"]: pipeline["name"] or f"Pipeline #{pipeline['id']}"
            for pipeline in pipelines
        }
        selected_pipeline_id = st.selectbox(
            "Select Pipeline to Evaluate",
            options=list(pipeline_name_lookup.keys()),
            format_func=lambda pid: pipeline_name_lookup[pid],
        )
    else:
        st.info("No pipelines available. Configure one in the Pipeline Configuration tab.")

    if st.button("Run Visual Evaluation"):
        if total_rows == 0:
            st.warning("Gold standard dataset is empty.")
        elif not pipeline_name_lookup:
            st.warning("Create a pipeline before running the evaluation.")
        elif not text_col:
            st.error("Gold standard dataset does not contain a report/text column.")
        elif not concept_cols:
            st.error("No concept columns detected for evaluation.")
        else:
            pipeline_record = next(
                (p for p in pipelines if p["id"] == selected_pipeline_id), None
            )
            if pipeline_record is None:
                st.error("Unable to load the selected pipeline.")
            else:
                runtime = hydrate_pipeline_runtime(pipeline_record)
                if not runtime["models"]:
                    st.error("The selected pipeline has no valid models configured.")
                else:
                    gold_pdf = gold_df.to_pandas().head(train_rows)
                    pipeline_name = pipeline_name_lookup[selected_pipeline_id]
                    st.session_state.gold_eval_error = ""
                    with st.spinner(f"Running evaluation with {pipeline_name}"):
                        try:
                            eval_outputs = run_pipeline_prediction(
                                gold_pdf.head(),
                                id_col=id_col,
                                text_col=text_col,
                                concept_cols=concept_cols,
                                model_confs=runtime["models"],
                                system_prompt=runtime["system_prompt"],
                                user_prompt=runtime["user_prompt"],
                                anti=runtime["anti"],
                                judge_conf=runtime["judge_model"],
                                max_concurrency=min(4, max(1, total_rows)),
                            )
                            metrics_df, summary = compute_concept_metrics(
                                gold_pdf, eval_outputs["aggregated"], concept_cols, id_col
                            )
                            per_model_payload = [
                                {
                                    "label": runtime["models"][idx].get("name")
                                    or runtime["models"][idx].get("model"),
                                    "data": model_df,
                                }
                                for idx, model_df in enumerate(eval_outputs["per_model"])
                            ]
                            st.session_state.gold_eval_result = {
                                "pipeline_name": pipeline_name,
                                "timestamp": datetime.now().isoformat(),
                                "predictions": eval_outputs["aggregated"],
                                "per_model": per_model_payload,
                                "judge": eval_outputs["judge"],
                                "metrics": metrics_df,
                                "summary": summary,
                                "train_split": train_rows,
                                "test_split": test_rows,
                                "train_pct": train_pct,
                            }
                        except Exception as exc:
                            st.session_state.gold_eval_error = str(exc)
                            st.session_state.gold_eval_result = None

    if st.session_state.gold_eval_error:
        st.error(f"Evaluation failed: {st.session_state.gold_eval_error}")

    result_payload = st.session_state.gold_eval_result
    if result_payload:
        st.subheader("Latest Pipeline Evaluation")
        summary = result_payload["summary"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("Precision", f"{summary['precision']:.2%}")
        metric_cols[1].metric("Recall", f"{summary['recall']:.2%}")
        metric_cols[2].metric("F1", f"{summary['f1']:.2%}")
        metric_cols[3].metric("Accuracy", f"{summary['accuracy']:.2%}")
        st.caption(
            f"{result_payload['pipeline_name']} · "
            f"{result_payload['train_split']} train / {result_payload['test_split']} test "
            f"({result_payload['train_pct']}% split)"
        )

        st.markdown("#### Per-Concept Metrics")
        metrics_view = result_payload["metrics"].copy()
        numeric_cols = ["precision", "recall", "f1", "accuracy"]
        for col in numeric_cols:
            if col in metrics_view.columns:
                metrics_view[col] = metrics_view[col].apply(
                    lambda v: f"{v:.2%}" if isinstance(v, (int, float)) else "—"
                )
        st.dataframe(metrics_view, use_container_width=True, hide_index=True)

        st.markdown("#### Predictions Preview")
        prediction_df = result_payload["predictions"]
        st.dataframe(prediction_df.head(50), use_container_width=True)
        csv_bytes = prediction_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv_bytes,
            file_name="pipeline_predictions.csv",
            mime="text/csv",
        )

        with st.expander("Per-model outputs"):
            for idx, payload in enumerate(result_payload["per_model"], start=1):
                st.markdown(f"**Model {idx}: {payload['label']}**")
                st.dataframe(payload["data"].head(20), use_container_width=True)

        if result_payload["judge"] is not None:
            with st.expander("Judge Model Output"):
                st.dataframe(result_payload["judge"].head(20), use_container_width=True)


def render_pipeline_evaluation_tab() -> None:
    """Orchestrate sidebar, management, and analysis sections."""
    annotations = st.session_state.annotations_db
    id_col, concept_cols = render_annotations_sidebar(annotations)
    tab1, tab2, tab3 = st.tabs(
        ["Annotations Management", "Inter-Annotator Analysis", "Gold Standard Evaluation"]
    )
    with tab1:
        render_annotations_management_tab()
    with tab2:
        render_inter_annotator_analysis_tab(id_col, concept_cols)
    with tab3:
        render_gold_standard_tab(id_col, concept_cols)


def main() -> None:
    ensure_session_state()
    sanitize_pipeline_state()

    tabs = st.tabs(["Pipeline Configuration", "Pipeline Unit Test", "Pipeline Evaluation"])
    with tabs[0]:
        pip_tabs = st.tabs(["Models", "Prompts", "Pipeline"])
        with pip_tabs[0]:
            render_models_tab()
        with pip_tabs[1]:
            render_prompts_tab()
        with pip_tabs[2]:
            render_pipeline_tab()
    with tabs[1]:
        ut_tabs = st.tabs(["Report Selection", "Extraction", "Raw Output"])
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
