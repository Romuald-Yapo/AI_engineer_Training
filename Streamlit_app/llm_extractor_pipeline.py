from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# ---------------- Core Functions ---------------- #
def load_model_ollama(model_name, num_ctx, max_output, temperature, top_p, top_k):
    return Ollama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        top_p=top_p,
        top_k=top_k,
        base_url="http://lx181.intra.chu-rennes.fr:11434/",
    )  # Hyperparameters of the model


def build_prompt(messages):
    return ChatPromptTemplate.from_messages(messages)


def initialise_chain(llm, prompt):
    return LLMChain(llm=llm, prompt=prompt)


def extract_json_block(s: str) -> str | None:
    start = s.find("[")
    if start == -1:
        return None
    brace_count = 0
    for i in range(start, len(s)):
        if s[i] == "[":
            brace_count += 1
        elif s[i] == "]":
            brace_count -= 1
            if brace_count == 0:
                return s[start : i + 1]
    return None


def process_cr_json(cr_text, chain):
    response = chain.invoke({"cr_medical": cr_text})
    raw_output = response["text"]

    output = extract_json_block(raw_output)

    if output is None:
        return None, raw_output

    try:
        if isinstance(output, str):
            return json.loads(output), raw_output
    except Exception:
        return None, raw_output

    return None, raw_output


def validate_format(data) -> bool:
    if not isinstance(data, list) or len(data) != 4:
        return False
    required_keys = {"concept", "context", "presence"}
    for obj in data:
        if not isinstance(obj, dict):
            return False
        if set(obj.keys()) != required_keys:
            return False
    return True


def build_messages(filtered_data, tech_prompt, user_prompt: str) -> list:
    cr_text = filtered_data.collect().to_pandas().iloc[0]["text"]

    messages = [SystemMessage(content=tech_prompt)]
    messages.append(HumanMessage(content=f"{user_prompt} \nLe Compte rendu mǸdical : {cr_text}"))
    return messages


def _normalize_presence_label(value):
    """Map presence values to booleans for comparison across LLMs."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "vrai", "present", "prǸsent", "oui", "yes", "1"}:
            return True
        if token in {"false", "faux", "absent", "non", "no", "0"}:
            return False
    return None


def _prepare_model_conf(model_name, num_ctx, max_output, temperature, top_p, top_k):
    """Normalize expected fields for model execution."""
    return {
        "model": model_name,
        "num_ctx": num_ctx,
        "max_output": max_output,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }


def _run_single_model(messages, cr_text, model_conf, max_retries=3):
    """Execute one model with retries and return structured outputs."""
    llm = load_model_ollama(
        model_conf["model"],
        model_conf.get("num_ctx"),
        model_conf.get("max_output"),
        model_conf.get("temperature"),
        model_conf.get("top_p"),
        model_conf.get("top_k"),
    )

    concept_extracted = None
    brut_response = ""
    local_messages = deepcopy(messages)

    for attempt in range(1, max_retries + 1):
        prompt = build_prompt(local_messages)
        chaine = initialise_chain(llm, prompt)

        concept_extracted, brut_response = process_cr_json(cr_text, chaine)

        if concept_extracted is not None and validate_format(concept_extracted):
            print(f"Format respectǸ �� la tentative {attempt}")
            return {
                "concept_extracted": concept_extracted,
                "brut_response": brut_response,
                "model": model_conf["model"],
                "attempts": attempt,
                "error": False,
            }

        local_messages.append(
            HumanMessage(content="Format de Sortie pas respectǸ. RESPECTE les instructions donnǸes.")
        )

    print(f"�%chec apr��s {max_retries} tentatives")
    return {
        "concept_extracted": concept_extracted if concept_extracted else "Erreur",
        "brut_response": brut_response,
        "model": model_conf["model"],
        "attempts": max_retries,
        "error": True,
    }


def _combine_ensemble_outputs(outputs: list):
    """Merge multiple JSON outputs and enforce agreement on presence labels."""
    combined = {}
    disagreement = False

    for output in outputs:
        if not isinstance(output, list):
            disagreement = True
            continue
        for item in output:
            concept = item.get("concept")
            if concept is None:
                continue
            normalized_presence = _normalize_presence_label(item.get("presence"))
            existing = combined.get(concept, {"presence": None, "contexts": set()})
            if existing["presence"] is not None and normalized_presence is not None:
                if existing["presence"] != normalized_presence:
                    disagreement = True
            if normalized_presence is not None and existing["presence"] is None:
                existing["presence"] = normalized_presence
            context_val = item.get("context")
            if context_val:
                existing["contexts"].add(context_val)
            combined[concept] = existing

    if disagreement:
        return None, "Les LLMs n'ont pas ǸtǸ d'accord sur la prǸsence des concepts."

    merged = []
    for concept, payload in combined.items():
        merged.append(
            {
                "concept": concept,
                "context": "; ".join(sorted(payload["contexts"])) if payload["contexts"] else "",
                "presence": payload["presence"] if payload["presence"] is not None else False,
            }
        )
    return merged, ""


# ---------------- Main Extractor ---------------- #
def llm_extractor(messages, filtered_data, model_name, num_ctx, max_output, temperature, top_p, top_k, max_retries=3):
    cr_text = filtered_data.collect().to_pandas().iloc[0]["text"]
    model_conf = _prepare_model_conf(model_name, num_ctx, max_output, temperature, top_p, top_k)
    result = _run_single_model(messages, cr_text, model_conf, max_retries=max_retries)

    return pd.DataFrame(
        [
            {
                "concept_extracted": result["concept_extracted"],
                "brut_response": result["brut_response"],
                "ensemble_notice": "",
            }
        ]
    )


def llm_ensemble_extractor(messages, filtered_data, model_confs, max_retries=3, ensemble_size=2):
    """Run multiple LLMs in parallel and merge outputs when they agree."""
    if not model_confs:
        raise ValueError("Aucun mod��le configurǸ pour l'ensemble.")

    cr_text = filtered_data.collect().to_pandas().iloc[0]["text"]
    normalized = []
    for conf in model_confs[:ensemble_size]:
        normalized.append(
            _prepare_model_conf(
                conf.get("model_version") or conf.get("model"),
                conf.get("num_ctx"),
                conf.get("max_output"),
                conf.get("temperature"),
                conf.get("top_p"),
                conf.get("top_k"),
            )
        )

    outputs = []
    with ThreadPoolExecutor(max_workers=len(normalized)) as executor:
        future_map = {
            executor.submit(_run_single_model, messages, cr_text, conf, max_retries): conf["model"]
            for conf in normalized
        }
        for future in as_completed(future_map):
            outputs.append(future.result())

    merged_json, notice = _combine_ensemble_outputs([res.get("concept_extracted") for res in outputs])
    raw_outputs = "\n\n---\n\n".join(
        f"{res.get('model', '')}:\n{res.get('brut_response', '')}" for res in outputs
    )

    return pd.DataFrame(
        [
            {
                "concept_extracted": merged_json if merged_json is not None else [],
                "brut_response": raw_outputs,
                "ensemble_notice": notice or "",
            }
        ]
    )
