"""
Explainability engine for the Fake Job Posting Detector.

Same leave-one-word-out logic used in the training notebook:
for each word, we ask "what does the model predict with this word removed?"
and compare that to the prediction with the word present. The difference
is that word's "impact" — positive pushes toward REAL, negative toward FAKE.
"""

import re
import numpy as np


def tokenize(text: str):
    """Splits text into word-like tokens (letters/numbers/apostrophes)."""
    return re.findall(r"[A-Za-z0-9']+", text)


def get_predict_fn(model, vectorizer):
    """
    Wraps a trained ML model + its TF-IDF vectorizer into a single
    function: list[str] -> np.array of P(FAKE).
    Falls back to a sigmoid-squashed decision_function for models
    (like plain LinearSVC) that don't expose predict_proba.
    """
    def predict_fn(texts):
        vec = vectorizer.transform(texts)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(vec)[:, 1]
        from scipy.special import expit
        return expit(model.decision_function(vec))
    return predict_fn


def word_impacts(text: str, predict_fn):
    """
    Leave-one-word-out analysis.
    Returns (baseline_prob_real, list of (word, impact)).
    impact > 0  -> word pushed the prediction toward REAL (green)
    impact < 0  -> word pushed the prediction toward FAKE (red)
    """
    tokens = tokenize(text)
    if not tokens:
        return 0.5, []

    baseline_fake = predict_fn([text])[0]
    baseline_real = 1 - baseline_fake

    impacts = []
    for i in range(len(tokens)):
        remaining = tokens[:i] + tokens[i + 1:]
        remaining_text = " ".join(remaining) if remaining else ""
        removed_fake = predict_fn([remaining_text])[0] if remaining_text else 0.5
        removed_real = 1 - removed_fake
        impact = baseline_real - removed_real
        impacts.append((tokens[i], float(impact)))

    return float(baseline_real), impacts


def flip_analysis(text: str, predict_fn, impacts, tokens, max_removals: int = 5):
    """
    Greedily removes the most influential words toward the CURRENT predicted
    label and checks whether that alone is enough to flip the prediction.
    A simple robustness / sanity check on how "fragile" the decision is.
    """
    if not tokens:
        return False, [], None

    baseline_fake = predict_fn([text])[0]
    current_label = "FAKE" if baseline_fake >= 0.5 else "REAL"

    if current_label == "FAKE":
        ranked = sorted(impacts, key=lambda t: t[1])          # most FAKE-pushing first
    else:
        ranked = sorted(impacts, key=lambda t: -t[1])         # most REAL-pushing first

    working = list(tokens)
    removed = []
    for word, _ in ranked[:max_removals]:
        if word in working:
            working.remove(word)
            removed.append(word)
        new_text = " ".join(working)
        new_fake = predict_fn([new_text])[0] if new_text else 0.5
        new_label = "FAKE" if new_fake >= 0.5 else "REAL"
        if new_label != current_label:
            return True, removed, float(new_fake)

    return False, removed, None


def analyze(title: str, description: str, model, vectorizer, max_words: int = 80):
    """
    Full explainable prediction. Returns a dict the Streamlit UI can render:
    baseline REAL probability, per-word impacts, top contributors, and
    the flip-analysis result.
    """
    predict_fn = get_predict_fn(model, vectorizer)

    text = f"{title} {description}".strip()
    tokens_full = tokenize(text)
    truncated = len(tokens_full) > max_words
    text_used = " ".join(tokens_full[:max_words]) if truncated else text

    baseline_real, impacts = word_impacts(text_used, predict_fn)
    tokens_used = tokenize(text_used)

    top_words = sorted(impacts, key=lambda t: -abs(t[1]))[:10]
    flipped, removed_words, new_fake_prob = flip_analysis(text_used, predict_fn, impacts, tokens_used)

    return {
        "text_used": text_used,
        "truncated": truncated,
        "baseline_prob_real": baseline_real,
        "baseline_prob_fake": 1 - baseline_real,
        "label": "REAL" if baseline_real >= 0.5 else "FAKE",
        "impacts": impacts,
        "top_words": top_words,
        "flipped": flipped,
        "removed_words": removed_words,
        "new_fake_prob": new_fake_prob,
    }
