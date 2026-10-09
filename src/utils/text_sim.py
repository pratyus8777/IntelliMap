import re
import math
import unicodedata
from difflib import SequenceMatcher
from collections import Counter
from typing import List, Set

_word = re.compile(r"[A-Za-z0-9]+")

def norm_name(s: str) -> str:
    """Legacy normalization: lower, replace non-word with spaces."""
    return re.sub(r"[_\W]+", " ", (s or "")).strip().lower()

def normalize_name(s: str) -> str:
    """Stronger normalization for ML: remove accents, non-alphanum, lowercase."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8", "ignore")
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s

def token_set(s: str) -> Set[str]:
    return set(_word.findall(norm_name(s)))

def jaccard(a: str, b: str) -> float:
    A, B = token_set(a), token_set(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def levenshtein_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_name(a), norm_name(b)).ratio()

def char_ngrams(s: str, n: int = 3) -> List[str]:
    s2 = f"^{norm_name(s)}$"
    return [s2[i:i+n] for i in range(max(1, len(s2)-n+1))]

def cosine_ngrams(a: str, b: str, n: int = 3) -> float:
    A, B = char_ngrams(a, n), char_ngrams(b, n)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    ca, cb = Counter(A), Counter(B)
    num = sum(ca[g]*cb.get(g,0) for g in ca)
    da = math.sqrt(sum(v*v for v in ca.values()))
    db = math.sqrt(sum(v*v for v in cb.values()))
    return (num/(da*db)) if da and db else 0.0

def fuzzy_name_score(a: str, b: str) -> float:
    """Blend of three string metrics (0..1)."""
    return 0.4 * levenshtein_ratio(a,b) + 0.3 * jaccard(a,b) + 0.3 * cosine_ngrams(a,b,3)

# backward-compatible alias
def string_similarity(a: str, b: str) -> float:
    return fuzzy_name_score(a,b)
