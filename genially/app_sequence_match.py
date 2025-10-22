import html
import random
import textwrap
from typing import Optional

import streamlit as st

from token_matcher import (
    MatchResult,
    compute_window_distances,
    find_sequence_in_text,
    tokenize_like_langextract,
)

st.set_page_config(page_title="Token Sequence Matcher", layout="wide")


EXAMPLES = [
    {
        "title": "Customer support chat",
        "full_text": textwrap.dedent(
            """
            Agent: Thanks for reaching out to us today. I understand the dashboard will not load on your end.
            Customer: Correct, it spins and never resolves.
            Agent: I can see an outage on the analytics cluster. As a workaround you can export the CSV summary.
            Customer: That should work for now, thank you.
            Agent: A fix will ship tomorrow morning.
            """
        ).strip(),
        "sequence_text": "it spins and never resolves",
        "threshold_ratio": 0.25,
    },
    {
        "title": "Product specification",
        "full_text": textwrap.dedent(
            """
            The alpha release focuses on fast authentication and resilient offline mode.
            Subsequent iterations will add collaborative editing and advanced analytics dashboards.
            A staged rollout is planned, starting with the beta testers in November.
            """
        ).strip(),
        "sequence_text": "collaborative editing and advanced analytics dashboards",
        "threshold_ratio": 0.0,
    },
    {
        "title": "Research notes",
        "full_text": textwrap.dedent(
            """
            Observed subjects respond positively to long-form guidance.
            They rarely follow terse instructions verbatim.
            Reinforcement in the form of visual aids increases compliance by roughly 18 percent.
            """
        ).strip(),
        "sequence_text": "respond positively to long form guidance",
        "threshold_ratio": 0.35,
    },
]


def load_example() -> None:
    sample = random.choice(EXAMPLES)
    st.session_state.full_text = sample["full_text"]
    st.session_state.sequence_text = sample["sequence_text"]
    st.session_state.threshold_ratio = sample["threshold_ratio"]
    st.toast(f"Loaded example: {sample['title']}")


def get_state_value(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def highlight_span(text: str, start: int, end: int) -> str:
    prefix = html.escape(text[:start])
    match = html.escape(text[start:end])
    suffix = html.escape(text[end:])
    return (
        "<div style='font-family:monospace; white-space:pre-wrap;'>"
        f"{prefix}<mark>{match}</mark>{suffix}"
        "</div>"
    )


st.title("Token Sequence Approximate Matcher")
st.caption(
    "Tokenises both inputs with a rule mirroring langextract, slides a window over the full text, "
    "and stops when the fuzzywuzzy similarity ratio falls within the chosen threshold."
)

col_gen, col_case = st.columns([1, 1])
with col_gen:
    if st.button("Generate example data"):
        load_example()
with col_case:
    case_sensitive = st.checkbox("Case sensitive comparison", value=False)

full_text = st.text_area(
    "Full text",
    get_state_value("full_text", EXAMPLES[0]["full_text"]),
    height=260,
)
sequence_text = st.text_area(
    "Sequence to locate",
    get_state_value("sequence_text", EXAMPLES[0]["sequence_text"]),
    height=140,
)

threshold_ratio = st.slider(
    "Minimum similarity ratio (0 = no match, 1 = exact)",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    value=float(get_state_value("threshold_ratio", EXAMPLES[0]["threshold_ratio"])),
)

execute = st.button("Run search", type="primary")

result: Optional[MatchResult] = None
if execute:
    st.session_state.threshold_ratio = threshold_ratio
    result = find_sequence_in_text(
        sequence_text,
        full_text,
        threshold_ratio=threshold_ratio,
        case_sensitive=case_sensitive,
    )

result_column, detail_column = st.columns([1, 1])
with result_column:
    if result:
        st.subheader("Matched span")
        st.markdown(highlight_span(full_text, result.start, result.end), unsafe_allow_html=True)
        st.metric("Similarity ratio", f"{result.ratio:.2f}")
        st.metric("Percentage", f"{result.ratio * 100:.0f}%")
        st.write(
            {
                "start_char": result.start,
                "end_char": result.end,
                "window_tokens": list(result.window_tokens),
                "query_tokens": list(result.query_tokens),
                "ratio": result.ratio,
            }
        )
    elif execute:
        st.warning("No window satisfied the similarity threshold. Lower the threshold or adjust the sequence.")

with detail_column:
    if execute:
        st.subheader("Token diagnostics")
        query_tokens = tokenize_like_langextract(sequence_text, lowercase=not case_sensitive)
        st.write(f"Sequence token count: {len(query_tokens)}")
        window_rows = compute_window_distances(
            sequence_text, full_text, case_sensitive=case_sensitive
        )
        if window_rows:
            st.dataframe(window_rows, use_container_width=True, hide_index=True)
        else:
            st.info("Full text does not contain enough tokens for the provided sequence.")
