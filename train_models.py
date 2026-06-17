"""
train_models.py
---------------
End-to-end training pipeline for the Social Media Emotion Analyzer (Theme 5).

What it does
============
1. Loads the emotion dataset (train / validation / test CSVs).
2. Cleans every text using the shared preprocess.clean_text pipeline.
3. Builds TWO feature representations:
      (a) TF-IDF  (word + bigram, sklearn)
      (b) Word2Vec averaged embeddings (gensim)
4. Trains and compares THREE classifiers:
      - Multinomial Naive Bayes  (TF-IDF only — needs non-negative features)
      - Logistic Regression       (both feature sets)
      - Linear SVM                (both feature sets)
5. Evaluates each on the held-out TEST set (accuracy, precision, recall, F1).
6. Saves:
      - models/tfidf_vectorizer.pkl
      - models/word2vec.model
      - models/best_model.pkl           (primary model used by the app)
      - models/all_results.json         (metrics for every model — for charts)
      - models/confusion_matrix.json    (for the best model)

Run with:  python train_models.py
"""

import os
import json
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)
from gensim.models import Word2Vec

from preprocess import clean_text, tokens, LABEL_MAP

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)

EMOTION_NAMES = [LABEL_MAP[i] for i in range(len(LABEL_MAP))]


def load_data():
    train = pd.read_csv(os.path.join(DATA, "training.csv"))
    val = pd.read_csv(os.path.join(DATA, "validation.csv"))
    test = pd.read_csv(os.path.join(DATA, "test.csv"))
    # Combine train + validation for final training (test stays untouched).
    full_train = pd.concat([train, val], ignore_index=True)
    return full_train, test


def document_vector(w2v, doc_tokens, dim):
    """Average the Word2Vec vectors of all in-vocabulary tokens in a doc."""
    vecs = [w2v.wv[t] for t in doc_tokens if t in w2v.wv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def evaluate(name, feature, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    print(f"  {name:<28} [{feature:<8}]  acc={acc:.4f}  f1={f1:.4f}")
    return {
        "model": name,
        "feature": feature,
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
    }


def main():
    t0 = time.time()
    print("Loading data...")
    train, test = load_data()
    print(f"  Train+Val: {len(train)} rows | Test: {len(test)} rows")

    print("Cleaning text (this takes a moment)...")
    train["clean"] = train["text"].apply(clean_text)
    test["clean"] = test["text"].apply(clean_text)

    y_train = train["label"].values
    y_test = test["label"].values

    results = []

    # ----------------------------------------------------------------- TF-IDF
    print("\n[Feature 1] TF-IDF (unigram + bigram, max_features=8000)")
    tfidf = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
    )
    Xtr_tfidf = tfidf.fit_transform(train["clean"])
    Xte_tfidf = tfidf.transform(test["clean"])

    # Naive Bayes (TF-IDF)
    nb = MultinomialNB()
    nb.fit(Xtr_tfidf, y_train)
    results.append(evaluate("Naive Bayes", "TF-IDF", y_test, nb.predict(Xte_tfidf)))

    # Logistic Regression (TF-IDF)
    lr = LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")
    lr.fit(Xtr_tfidf, y_train)
    lr_pred = lr.predict(Xte_tfidf)
    results.append(evaluate("Logistic Regression", "TF-IDF", y_test, lr_pred))

    # Linear SVM (TF-IDF)
    svm = LinearSVC(C=1.0, class_weight="balanced")
    svm.fit(Xtr_tfidf, y_train)
    results.append(evaluate("Linear SVM", "TF-IDF", y_test, svm.predict(Xte_tfidf)))

    # --------------------------------------------------------------- Word2Vec
    print("\n[Feature 2] Word2Vec (100-dim, averaged embeddings)")
    train_tokens = [tokens(t) for t in train["text"]]
    test_tokens = [tokens(t) for t in test["text"]]
    DIM = 100
    w2v = Word2Vec(
        sentences=train_tokens, vector_size=DIM, window=5,
        min_count=2, workers=4, epochs=20, seed=42,
    )
    Xtr_w2v = np.vstack([document_vector(w2v, d, DIM) for d in train_tokens])
    Xte_w2v = np.vstack([document_vector(w2v, d, DIM) for d in test_tokens])

    # Logistic Regression (Word2Vec)
    lr_w = LogisticRegression(max_iter=1000, C=5.0, class_weight="balanced")
    lr_w.fit(Xtr_w2v, y_train)
    results.append(evaluate("Logistic Regression", "Word2Vec", y_test, lr_w.predict(Xte_w2v)))

    # Linear SVM (Word2Vec)
    svm_w = LinearSVC(C=1.0, class_weight="balanced")
    svm_w.fit(Xtr_w2v, y_train)
    results.append(evaluate("Linear SVM", "Word2Vec", y_test, svm_w.predict(Xte_w2v)))

    # ----------------------------------------------------- pick the best model
    # The app needs probability estimates for confidence scores, so among the
    # top performers we prefer Logistic Regression on TF-IDF (predict_proba).
    best = max(results, key=lambda r: r["f1"])
    print(f"\nBest overall: {best['model']} [{best['feature']}] (F1={best['f1']})")

    # The deployable model for the app = Logistic Regression on TF-IDF
    # (strong accuracy AND calibrated probabilities for confidence bars).
    app_pred = lr_pred
    cm = confusion_matrix(y_test, app_pred).tolist()

    print("\nClassification report (Logistic Regression + TF-IDF — used in app):")
    print(classification_report(y_test, app_pred, target_names=EMOTION_NAMES, zero_division=0))

    # ----------------------------------------------------------------- save
    joblib.dump(tfidf, os.path.join(MODELS, "tfidf_vectorizer.pkl"))
    joblib.dump(lr, os.path.join(MODELS, "best_model.pkl"))
    w2v.save(os.path.join(MODELS, "word2vec.model"))

    with open(os.path.join(MODELS, "all_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(MODELS, "confusion_matrix.json"), "w") as f:
        json.dump({"matrix": cm, "labels": EMOTION_NAMES}, f, indent=2)

    # App-facing metrics summary
    app_metrics = next(r for r in results
                       if r["model"] == "Logistic Regression" and r["feature"] == "TF-IDF")
    with open(os.path.join(MODELS, "app_metrics.json"), "w") as f:
        json.dump(app_metrics, f, indent=2)

    print(f"\nAll artifacts saved to /models. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
