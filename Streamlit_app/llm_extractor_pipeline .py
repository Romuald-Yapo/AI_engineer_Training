from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from textwrap import dedent

# ---------------- Core Functions ---------------- #
def load_model_ollama(model_name, num_ctx, max_output, temperature, top_p, top_k):
    return Ollama(
        model=model_name,
        temperature=temperature,
        num_ctx=num_ctx,
        top_p=top_p,
        top_k=top_k
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

def _extract_system_prompt(messages: list) -> str:
    """Return the first system prompt content if available."""
    for msg in messages:
        if isinstance(msg, SystemMessage):
            return msg.content
    return ""


def _extract_user_prompt(messages: list) -> str:
    """Return the first user/clinician instruction message content."""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _run_prompt_for_json(llm, messages, cr_text):
    """Execute a prompt expecting JSON output and return parsed data plus the raw text."""
    prompt = build_prompt(messages)
    chain = initialise_chain(llm, prompt)
    return process_cr_json(cr_text, chain)


def _run_prompt_for_text(llm, messages, cr_text) -> str:
    """Execute a prompt and return the raw text response."""
    prompt = build_prompt(messages)
    chain = initialise_chain(llm, prompt)
    return chain.invoke({"cr_medical": cr_text})['text']


def _coerce_text_output(response) -> str:
    """Return a plain string regardless of the LLM response type."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response.get("text", ""))
    content = getattr(response, "content", None)
    if content is not None:
        return content
    return str(response)


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


def _attempt_json_conversation(llm, messages, cr_text, max_retries=3):
    """Retry a given prompt until a valid JSON payload complying with the schema is produced."""
    concept_extracted = None
    raw_output = ""
    local_messages = deepcopy(messages)

    for attempt in range(1, max_retries + 1):
        concept_extracted, raw_output = _run_prompt_for_json(llm, local_messages, cr_text)
        if concept_extracted is not None and validate_format(concept_extracted):
            return concept_extracted, raw_output, attempt, False

        local_messages.append(
            HumanMessage(content="Format de Sortie pas respect��. RESPECTE les instructions donn��es.")
        )

    fallback = concept_extracted if concept_extracted else "Erreur"
    return fallback, raw_output, max_retries, True


def _build_verification_prompt() -> str:
    """Create the follow-up question that double checks the format and captured contexts."""
    return dedent(
        f"""
        réponds aux deux questions ci-dessous à partir du compte rendu initial:


        1. Le format demandé (liste JSON d'objets {{concept, context, presence}}) est-il strictement respecté ?
           Réponds par Oui ou Non et explique ce qui manque si nécessaire.
        2. Chaque concept contient-il le bon contexte complet et toutes les informations utiles compte tenu du rapport ?
           Réponds par Oui ou Non et détaille précisément les corrections à apporter.

        Présente ta réponse comme suit :
        Format_OK: Oui/Non - courte justification
        Context_OK: Oui/Non - courte justification
        Corrections: liste claire des ajustements nécessaires
        """
    ).strip()


def _build_reproduction_prompt(verification_feedback: str) -> str:
    """Create the final instruction asking for a corrected JSON output."""
    return dedent(
        f"""
        Ton analyse de vérification a produit :
        {verification_feedback}

        En te basant sur ces remarques, reprends entièrement l'extraction.

        Produit UNIQUEMENT le JSON final, strictement au format attendu (liste d'objets {{concept, context, presence}}),
        en appliquant toutes les corrections mentionnées. N'inclus aucun texte supplémentaire.
        """
    ).strip()


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

    concept_extracted, brut_response, attempts, error = _attempt_json_conversation(
        llm, messages, cr_text, max_retries=max_retries
    )

    if not error:
        print(f"Format respectǸ �� la tentative {attempts}")
    else:
        print(f"�%chec apr��s {max_retries} tentatives")

    return {
        "concept_extracted": concept_extracted,
        "brut_response": brut_response,
        "model": model_conf["model"],
        "attempts": attempts,
        "error": error,
    }


def _run_chain_of_verification(messages, cr_text, model_conf, max_retries=3):
    """Run a three-step conversation with history to verify and reproduce the JSON output."""
    llm = load_model_ollama(
        model_conf["model"],
        model_conf.get("num_ctx"),
        model_conf.get("max_output"),
        model_conf.get("temperature"),
        model_conf.get("top_p"),
        model_conf.get("top_k"),
    )

    system_prompt = _extract_system_prompt(messages) or "Tu es un assistant clinique rigoureux."
    user_prompt = _extract_user_prompt(messages)

    base_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    history = InMemoryChatMessageHistory()
    conversation = RunnableWithMessageHistory(
        base_prompt | llm,
        lambda _: history,
        input_messages_key="input",
        history_messages_key="history",
    )
    session_cfg = {"configurable": {"session_id": "chain_of_verification"}}

    logs: list[str] = []

    # Step 1: initial extraction identical to the standard user prompt.
    initial_instruction = user_prompt or dedent(
        f"""
        Analyse le compte rendu suivant et retourne uniquement la liste JSON demandée.
        Compte rendu clinique :
        {cr_text}
        """
    ).strip()
    initial_raw = _coerce_text_output(
        conversation.invoke({"input": initial_instruction}, config=session_cfg)
    ).strip()
    initial_json = extract_json_block(initial_raw)
    initial_data = None
    if initial_json:
        try:
            initial_data = json.loads(initial_json)
        except Exception:
            initial_data = None
    initial_valid = validate_format(initial_data) if isinstance(initial_data, list) else False
    logs.append(f"[Étape 1] Extraction initiale:\n{initial_raw}")

    if not initial_data:
        initial_data = []

    # Step 2: questioning about the content and the format.
    verification_prompt = _build_verification_prompt()
    verification_raw = _coerce_text_output(
        conversation.invoke({"input": verification_prompt}, config=session_cfg)
    ).strip()
    logs.append(f"[Étape 2] Contrôle format & contenu:\n{verification_raw}")

    # Step 3: reproduce the JSON output according to the answers.
    reproduction_prompt = _build_reproduction_prompt(verification_raw)
    final_raw = _coerce_text_output(
        conversation.invoke({"input": reproduction_prompt}, config=session_cfg)
    ).strip()
    final_json = extract_json_block(final_raw)
    final_data = None
    if final_json:
        try:
            final_data = json.loads(final_json)
        except Exception:
            final_data = None
    final_valid = validate_format(final_data) if isinstance(final_data, list) else False
    logs.append(f"[Étape 3] Reproduction finale:\n{final_raw}")

    payload = final_data if final_valid else (initial_data if initial_valid else "Erreur")
    error_flag = not final_valid

    return {
        "concept_extracted": payload,
        "brut_response": "\n\n".join(logs),
        "model": model_conf["model"],
        "attempts": 3,
        "error": error_flag,
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
def llm_extractor(
    messages,
    filtered_data,
    model_name,
    num_ctx,
    max_output,
    temperature,
    top_p,
    top_k,
    max_retries=3,
    chain_of_verification=False,
):
    cr_text = filtered_data.collect().to_pandas().iloc[0]["text"]
    model_conf = _prepare_model_conf(model_name, num_ctx, max_output, temperature, top_p, top_k)
    if chain_of_verification:
        result = _run_chain_of_verification(messages, cr_text, model_conf, max_retries=max_retries)
    else:
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
