import sqlite3
from copy import deepcopy
from pathlib import Path

import streamlit as st

from initialize_pipeline_db import DB_FILENAME, SCHEMA, ensure_database

st.set_page_config(page_title="LLM Medical Pipeline Configuration", layout="wide")

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
    "ensemble_size": 3,
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
        ensemble_size = max(ensemble_size, 2)

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
    state.pipeline_form_ensemble_size = int(payload["ensemble_size"]) if payload["ensemble_size"] else 3
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
        ensemble_size = max(ensemble_size, 2)
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
                st.text_input(
                    "Model identifier",
                    key="model_form_model",
                    help="Exact Ollama model tag, e.g. `mistral:latest`.",
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
                cols[0].write(
                    f"- Identifier: `{identifier}`\n"
                    f"- Temperature: {temperature_display}\n"
                    f"- Top P: {top_p_display}\n"
                    f"- Top K: {top_k_display}\n"
                    f"- Context window (num_ctx): {num_ctx_display}\n"
                    f"- Max output tokens: {max_output_display}"
                )
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
                cols[0].code(prompt["content"], language="markdown")
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
        st.slider("Ensemble size", 2, 10, key="pipeline_form_ensemble_size")

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

    if st.session_state.show_pipeline_form:
        render_pipeline_form()

    preview = get_pipeline_preview()
    if preview:
        st.divider()
        render_pipeline_summary(preview)


def main() -> None:
    ensure_session_state()
    sanitize_pipeline_state()

    st.title("LLM Medical Pipeline Configuration")
    st.write("Interactive workspace to configure models, prompts, and pipelines with anti-hallucination strategies.")

    tabs = st.tabs(["Models", "Prompts", "Pipeline"])
    with tabs[0]:
        render_models_tab()
    with tabs[1]:
        render_prompts_tab()
    with tabs[2]:
        render_pipeline_tab()


if __name__ == "__main__":
    main()
