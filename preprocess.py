"""
preprocess.py
-------------
Shared text-preprocessing utilities for the Social Media Emotion Analyzer.

Both the training pipeline (train_models.py / the notebook) and the Streamlit
app import from this module, so that the EXACT same cleaning steps are applied
at training time and at prediction time. This consistency is what makes the
saved models behave correctly inside the web app.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# Ensure required NLTK resources are present (safe to call repeatedly).
# ---------------------------------------------------------------------------
for _pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

# Emotion label mapping for the HuggingFace "emotion" dataset.
LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}

# Emoji + colour used everywhere in the UI for each emotion.
EMOTION_META = {
    "joy":      {"emoji": "😊", "color": "#FFB627"},
    "sadness":  {"emoji": "😢", "color": "#5B8DEF"},
    "love":     {"emoji": "❤️", "color": "#FF5D8F"},
    "anger":    {"emoji": "😠", "color": "#FF4D4D"},
    "fear":     {"emoji": "😨", "color": "#9B6DFF"},
    "surprise": {"emoji": "😲", "color": "#2EC4B6"},
}

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS = set(stopwords.words("english"))
# Keep negation words — they carry emotional signal ("not happy").
_NEGATIONS = {"no", "nor", "not", "never", "none", "cannot"}
_STOPWORDS = _STOPWORDS - _NEGATIONS

_URL_RE = re.compile(r"http\S+|www\.\S+")
_NONALPHA_RE = re.compile(r"[^a-z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply the full preprocessing pipeline to a single string.

    Steps: lowercase -> strip URLs -> remove non-alphabetic chars ->
    tokenize -> remove stopwords (keeping negations) -> lemmatize.
    Returns a cleaned, space-joined string.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NONALPHA_RE.sub(" ", text)
    tokens = _MULTISPACE_RE.sub(" ", text).strip().split()

    cleaned = [
        _LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in _STOPWORDS and len(tok) > 1
    ]
    return " ".join(cleaned)


def tokens(text: str) -> list:
    """Return the cleaned tokens as a list (used by Word2Vec averaging)."""
    return clean_text(text).split()
