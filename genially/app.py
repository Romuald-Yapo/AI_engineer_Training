import streamlit as st
from copy import deepcopy

st.set_page_config(page_title="LLM Medical Pipeline Configuration", layout="wide")

MODEL_DEFAULTS = {
    "name": "",
    "description": "",
    "provider": "anthropic",
    "model": "",
    "temperature": 0.7,
    "max_tokens": 2048,
    "top_p": 1.0,
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


def reset_model_form_fields(state: st.session_state, data: dict | None = None) -> None:
    payload = {**MODEL_DEFAULTS, **(data or {})}
    state.model_form_name = payload["name"]
    state.model_form_description = payload["description"]
    state.model_form_provider = payload["provider"]
    state.model_form_model = payload["model"]
    state.model_form_temperature = float(payload["temperature"])
    state.model_form_max_tokens = int(payload["max_tokens"]) if payload["max_tokens"] else 1
    state.model_form_top_p = float(payload["top_p"])


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

    if "models" not in state:
        state.models = []
    if "model_id_counter" not in state:
        state.model_id_counter = 1
    if "model_edit_id" not in state:
        state.model_edit_id = None
    if "show_model_form" not in state:
        state.show_model_form = False
    if "model_form_error" not in state:
        state.model_form_error = ""
    if "model_form_name" not in state:
        reset_model_form_fields(state)

    if "prompts" not in state:
        state.prompts = []
    if "prompt_id_counter" not in state:
        state.prompt_id_counter = 1
    if "prompt_edit_id" not in state:
        state.prompt_edit_id = None
    if "show_prompt_form" not in state:
        state.show_prompt_form = False
    if "prompt_form_error" not in state:
        state.prompt_form_error = ""
    if "prompt_form_name" not in state:
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
        pipeline_copy["anti_hallucination"] = anti
        updated_pipelines.append(pipeline_copy)
    state.pipelines = updated_pipelines

    if state.current_pipeline_id is not None:
        if not any(p["id"] == state.current_pipeline_id for p in state.pipelines):
            state.current_pipeline_id = None


def get_model_label(model_id: int) -> str:
    for model in st.session_state.models:
        if model["id"] == model_id:
            return f"{model['name']} ({model['provider']}/{model['model'] or 'n/a'})"
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
    state.models = [m for m in state.models if m["id"] != model_id]
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
        "provider": state.model_form_provider,
        "model": state.model_form_model.strip(),
        "temperature": float(state.model_form_temperature),
        "max_tokens": int(state.model_form_max_tokens),
        "top_p": float(state.model_form_top_p),
    }


def save_model_form() -> None:
    state = st.session_state
    data = collect_model_form_data()
    if not data["name"]:
        state.model_form_error = "Model name is required."
        return
    state.model_form_error = ""

    if state.model_edit_id is not None:
        updated = {"id": state.model_edit_id, **data}
        state.models = [
            updated if model["id"] == state.model_edit_id else model for model in state.models
        ]
    else:
        new_model = {"id": state.model_id_counter, **data}
        state.model_id_counter += 1
        state.models.append(new_model)

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
    state.prompts = [p for p in state.prompts if p["id"] != prompt_id]
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

    if state.prompt_edit_id is not None:
        updated = {"id": state.prompt_edit_id, **data}
        state.prompts = [
            updated if prompt["id"] == state.prompt_edit_id else prompt for prompt in state.prompts
        ]
    else:
        new_prompt = {"id": state.prompt_id_counter, **data}
        state.prompt_id_counter += 1
        state.prompts.append(new_prompt)

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
    state.pipelines = [p for p in state.pipelines if p["id"] != pipeline_id]
    if state.current_pipeline_id == pipeline_id:
        state.current_pipeline_id = None
    if state.pipeline_edit_id == pipeline_id:
        reset_pipeline_form()


def collect_pipeline_form_data(include_id: bool = True) -> dict:
    state = st.session_state
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
            "ensemble": bool(state.pipeline_form_ensemble),
            "ensembleSize": int(state.pipeline_form_ensemble_size),
            "chainOfVerification": bool(state.pipeline_form_chain_of_verification),
            "contextualGrounding": bool(state.pipeline_form_contextual_grounding),
            "llmAsJudge": bool(state.pipeline_form_llm_as_judge),
            "judgeModel": (
                int(state.pipeline_form_judge_model)
                if state.pipeline_form_llm_as_judge and state.pipeline_form_judge_model is not None
                else None
            ),
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
            "ensemble": anti.get("ensemble", False) or anti.get("ensemble", False),
            "ensemble_size": anti.get("ensembleSize", anti.get("ensemble_size", 3)),
            "chain_of_verification": anti.get("chainOfVerification", anti.get("chain_of_verification", False)),
            "contextual_grounding": anti.get("contextualGrounding", anti.get("contextual_grounding", False)),
            "llm_as_judge": anti.get("llmAsJudge", anti.get("llm_as_judge", False)),
            "judge_model": anti.get("judgeModel", anti.get("judge_model")),
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

    if state.pipeline_edit_id is not None:
        pipeline["id"] = state.pipeline_edit_id
        state.pipelines = [
            pipeline if existing["id"] == state.pipeline_edit_id else existing
            for existing in state.pipelines
        ]
    else:
        pipeline["id"] = state.pipeline_id_counter
        state.pipeline_id_counter += 1
        state.pipelines.append(pipeline)

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

    if st.session_state.show_model_form:
        header = "Edit model" if st.session_state.model_edit_id is not None else "New model"
        with st.container():
            st.markdown(f"### {header}")
            form_cols = st.columns(2)
            with form_cols[0]:
                st.text_input("Name", key="model_form_name")
            with form_cols[1]:
                st.selectbox("Provider", ["anthropic", "ollama", "openai"], key="model_form_provider")

            st.text_area(
                "Description",
                key="model_form_description",
                help="Short reminder about latency, pricing, or intended usage.",
            )

            spec_cols = st.columns(2)
            with spec_cols[0]:
                st.text_input("Model identifier", key="model_form_model")
                st.number_input("Max tokens", min_value=1, key="model_form_max_tokens", step=1)
            with spec_cols[1]:
                st.slider("Temperature", 0.0, 2.0, key="model_form_temperature", step=0.1)
                st.slider("Top P", 0.0, 1.0, key="model_form_top_p", step=0.05)

            if st.session_state.model_form_error:
                st.error(st.session_state.model_form_error)

            col1, col2 = st.columns(2)
            col1.button("Save model", on_click=save_model_form, key="model_form_save", type="primary", use_container_width=True)
            col2.button("Cancel", on_click=cancel_model_form, key="model_form_cancel", use_container_width=True)

    if st.session_state.models:
        st.markdown("### Saved models")
        for model in st.session_state.models:
            with st.container():
                cols = st.columns([6, 1, 1])
                cols[0].markdown(f"**{model['name']}**  \n{model['description'] or 'No description'}")
                cols[0].write(
                    f"- Provider: `{model['provider']}`\n"
                    f"- Identifier: `{model['model'] or 'n/a'}`\n"
                    f"- Temperature: {model['temperature']}\n"
                    f"- Top P: {model['top_p']}\n"
                    f"- Max tokens: {model['max_tokens']}"
                )
                cols[1].button("Edit", key=f"edit_model_{model['id']}", on_click=edit_model, args=(model["id"],))
                cols[2].button("Delete", key=f"delete_model_{model['id']}", on_click=delete_model, args=(model["id"],))
    else:
        st.info("No models configured yet. Use *New model* to add one.")


def render_prompts_tab() -> None:
    header_cols = st.columns([3, 1])
    with header_cols[0]:
        st.subheader("Prompt Configuration")
        st.write("Store reusable prompt templates for clinical extraction and technical formatting.")
    with header_cols[1]:
        st.button("New prompt", on_click=start_new_prompt, type="primary", use_container_width=True)

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
        for prompt in st.session_state.prompts:
            with st.container():
                cols = st.columns([6, 1, 1])
                badge = "Clinical" if prompt["type"] == "clinical" else "Technical"
                cols[0].markdown(f"**{prompt['name']}** · {badge}")
                cols[0].write(prompt["description"] or "No description")
                cols[0].code(prompt["content"], language="markdown")
                cols[1].button("Edit", key=f"edit_prompt_{prompt['id']}", on_click=edit_prompt, args=(prompt["id"],))
                cols[2].button("Delete", key=f"delete_prompt_{prompt['id']}", on_click=delete_prompt, args=(prompt["id"],))
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

    detail_cols = st.columns(2)
    with detail_cols[0]:
        st.write(f"**Name:** {pipeline['name'] or 'Unnamed pipeline'}")
        st.write(f"**Description:** {pipeline['description'] or 'No description'}")
    with detail_cols[1]:
        st.write(f"**Data source:** {pipeline['data_source'] or 'Not specified'}")

    st.markdown("#### Prompts")
    prompt_cols = st.columns(2)
    with prompt_cols[0]:
        st.markdown("**Clinical**")
        st.write(get_prompt_label(pipeline["clinical_prompt"]) if pipeline["clinical_prompt"] else "None")
    with prompt_cols[1]:
        st.markdown("**Technical**")
        st.write(get_prompt_label(pipeline["technical_prompt"]) if pipeline["technical_prompt"] else "None")

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

    if st.session_state.pipelines:
        st.markdown("### Saved pipelines")
        for pipeline in st.session_state.pipelines:
            with st.container():
                cols = st.columns([5, 1, 1, 1])
                cols[0].markdown(f"**{pipeline['name']}**  \n{pipeline['description'] or 'No description'}")
                cols[0].write(
                    f"- Models: {len(pipeline['selected_models'])}  \n"
                    f"- Strategies: {sum(1 for flag in pipeline['anti_hallucination'].values() if isinstance(flag, bool) and flag)}"
                )
                cols[1].button("Load", key=f"load_pipeline_{pipeline['id']}", on_click=load_pipeline, args=(pipeline["id"],))
                cols[2].button("Edit", key=f"edit_pipeline_{pipeline['id']}", on_click=edit_pipeline, args=(pipeline["id"],))
                cols[3].button("Delete", key=f"delete_pipeline_{pipeline['id']}", on_click=delete_pipeline, args=(pipeline["id"],))
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
