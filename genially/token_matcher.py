import dataclasses
import re
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    from fuzzywuzzy import fuzz
except ImportError as exc:  # pragma: no cover - helpful runtime guard
    raise ImportError(
        "fuzzywuzzy is required for token_matcher. Install it with `pip install fuzzywuzzy`."
    ) from exc


_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclasses.dataclass(frozen=True)
class Token:
    text: str
    normalised: str
    start: int
    end: int


@dataclasses.dataclass(frozen=True)
class MatchResult:
    matched_text: str
    ratio: float
    start: int
    end: int
    window_token_index: Tuple[int, int]
    query_tokens: Tuple[str, ...]
    window_tokens: Tuple[str, ...]


def tokenize_like_langextract(text: str, *, lowercase: bool = True) -> List[Token]:
    """
    Tokenise text with a rule-based scheme similar to langextract's regex tokenizer.
    Words, numbers, and punctuation marks are returned as distinct tokens.
    """
    normalise = str.lower if lowercase else (lambda s: s)
    tokens: List[Token] = []
    for match in _TOKEN_PATTERN.finditer(text):
        token_text = match.group(0)
        normalised = normalise(token_text)
        tokens.append(Token(token_text, normalised, match.start(), match.end()))
    return tokens


def find_sequence_in_text(
    sequence_text: str,
    full_text: str,
    *,
    threshold_ratio: Optional[float] = 0.0,
    case_sensitive: bool = False,
) -> Optional[MatchResult]:
    """
    Slide a token window across full_text until the fuzzywuzzy similarity ratio
    to sequence_text meets or exceeds threshold_ratio. Returns the first window
    satisfying the constraint, or the strongest match if threshold_ratio is None.
    """
    if sequence_text == "":
        return MatchResult("", 1.0, 0, 0, (0, 0), tuple(), tuple())

    sequence_tokens = tokenize_like_langextract(sequence_text, lowercase=not case_sensitive)
    full_tokens = tokenize_like_langextract(full_text, lowercase=not case_sensitive)

    if not sequence_tokens or not full_tokens:
        return None

    window_size = len(sequence_tokens)
    if window_size > len(full_tokens):
        return None

    if threshold_ratio is not None and not 0.0 <= threshold_ratio <= 1.0:
        raise ValueError("threshold_ratio must be between 0 and 1 (inclusive) or None.")

    sequence_norm = tuple(token.normalised for token in sequence_tokens)
    sequence_joined = " ".join(sequence_norm)

    best_ratio: Optional[float] = None
    best_window: Optional[Tuple[int, int]] = None
    best_window_tokens: Optional[Sequence[Token]] = None

    for start_index in range(0, len(full_tokens) - window_size + 1):
        end_index = start_index + window_size
        window = full_tokens[start_index:end_index]
        window_norm = tuple(token.normalised for token in window)
        window_joined = " ".join(window_norm)

        ratio = fuzz.ratio(sequence_joined, window_joined) / 100.0

        if threshold_ratio is not None and ratio >= threshold_ratio:
            return _build_result(
                full_text,
                window,
                sequence_norm,
                window_norm,
                ratio,
                start_index,
                end_index,
            )

        if best_ratio is None or ratio > best_ratio:
            best_ratio = ratio
            best_window = (start_index, end_index)
            best_window_tokens = window

    if (
        threshold_ratio is None
        and best_window is not None
        and best_ratio is not None
        and best_window_tokens is not None
    ):
        start_index, end_index = best_window
        window_norm = tuple(token.normalised for token in best_window_tokens)
        return _build_result(
            full_text,
            best_window_tokens,
            sequence_norm,
            window_norm,
            best_ratio,
            start_index,
            end_index,
        )

    return None


def _build_result(
    original_text: str,
    window_tokens: Sequence[Token],
    query_norm: Sequence[str],
    window_norm: Sequence[str],
    ratio: float,
    start_index: int,
    end_index: int,
) -> MatchResult:
    start_char = window_tokens[0].start
    end_char = window_tokens[-1].end
    matched_text = original_text[start_char:end_char]
    return MatchResult(
        matched_text=matched_text,
        ratio=ratio,
        start=start_char,
        end=end_char,
        window_token_index=(start_index, end_index),
        query_tokens=tuple(query_norm),
        window_tokens=tuple(window_norm),
    )


def stream_windows(tokens: Sequence[Token], size: int) -> Iterable[Tuple[int, int, Sequence[Token]]]:
    """
    Utility generator to stream token windows for debugging or visualisation.
    """
    if size <= 0:
        return
    upper_bound = len(tokens) - size + 1
    if upper_bound < 1:
        return
    for start in range(upper_bound):
        end = start + size
        yield start, end, tokens[start:end]


def compute_window_distances(
    sequence_text: str,
    full_text: str,
    *,
    case_sensitive: bool = False,
) -> List[dict]:
    """
    Return fuzzywuzzy similarity ratios for every token window matching sequence length.
    Useful for diagnostics and visualisation.
    """
    sequence_tokens = tokenize_like_langextract(sequence_text, lowercase=not case_sensitive)
    full_tokens = tokenize_like_langextract(full_text, lowercase=not case_sensitive)

    size = len(sequence_tokens)
    if size == 0 or size > len(full_tokens):
        return []

    sequence_norm = tuple(token.normalised for token in sequence_tokens)
    sequence_joined = " ".join(sequence_norm)

    rows: List[dict] = []
    for start, end, window in stream_windows(full_tokens, size):
        window_norm = tuple(token.normalised for token in window)
        window_joined = " ".join(window_norm)
        ratio = fuzz.ratio(sequence_joined, window_joined) / 100.0
        rows.append(
            {
                "start_token": start,
                "end_token": end,
                "ratio": ratio,
                "ratio_percent": round(ratio * 100, 2),
                "window_text": full_text[window[0].start : window[-1].end],
                "window_start": window[0].start,
                "window_end": window[-1].end,
                "window_tokens": window_norm,
            }
        )
    return rows
