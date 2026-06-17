"""
make_visuals.py
---------------
Generates the static visualization images used in the technical report and
poster. Saved into reports/figures/. (The Streamlit app generates its own
interactive versions live.)

Produces:
  1. class_distribution.png   - emotion counts in the dataset
  2. text_length.png          - distribution of message lengths
  3. model_comparison.png     - accuracy/F1 across all models
  4. confusion_matrix.png     - best model confusion matrix
  5. wordcloud_<emotion>.png  - one word cloud per emotion
  6. wordcloud_grid.png       - all six word clouds in one figure
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from preprocess import clean_text, LABEL_MAP, EMOTION_META

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")
FIG = os.path.join(HERE, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"
EMOTIONS = [LABEL_MAP[i] for i in range(6)]
COLORS = [EMOTION_META[e]["color"] for e in EMOTIONS]


def load():
    train = pd.read_csv(os.path.join(DATA, "training.csv"))
    train["emotion"] = train["label"].map(LABEL_MAP)
    return train


def fig1_distribution(df):
    counts = df["emotion"].value_counts().reindex(EMOTIONS)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=COLORS, edgecolor="white", linewidth=2)
    ax.set_title("Emotion Class Distribution (Training Set)", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel("Number of samples", fontsize=11)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v + 40, f"{v:,}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig2_length(df):
    df["wlen"] = df["text"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(9, 5))
    for e in EMOTIONS:
        sns.kdeplot(df[df.emotion == e]["wlen"], ax=ax, label=e,
                    color=EMOTION_META[e]["color"], fill=False, linewidth=2)
    ax.set_title("Message Length Distribution by Emotion", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Words per message", fontsize=11)
    ax.set_xlim(0, 60)
    ax.legend(title="Emotion")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "text_length.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig3_model_comparison():
    with open(os.path.join(MODELS, "all_results.json")) as f:
        results = json.load(f)
    labels = [f"{r['model']}\n({r['feature']})" for r in results]
    acc = [r["accuracy"] for r in results]
    f1 = [r["f1"] for r in results]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#6C5CE7", edgecolor="white")
    b2 = ax.bar(x + w/2, f1, w, label="F1-score", color="#00B894", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison", fontsize=15, fontweight="bold", pad=15)
    ax.legend()
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
                    f"{b.get_height():.2f}", ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "model_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig4_confusion():
    with open(os.path.join(MODELS, "confusion_matrix.json")) as f:
        cm_data = json.load(f)
    cm = np.array(cm_data["matrix"])
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="rocket_r",
                xticklabels=cm_data["labels"], yticklabels=cm_data["labels"],
                cbar_kws={"label": "Proportion"}, ax=ax, linewidths=1, linecolor="white")
    ax.set_title("Confusion Matrix — Logistic Regression + TF-IDF", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted emotion", fontsize=11)
    ax.set_ylabel("True emotion", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig5_wordclouds(df):
    df["clean"] = df["text"].apply(clean_text)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, e in zip(axes.ravel(), EMOTIONS):
        text = " ".join(df[df.emotion == e]["clean"])
        wc = WordCloud(width=500, height=350, background_color="white",
                       colormap="viridis", max_words=60).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{EMOTION_META[e]['emoji']}  {e.capitalize()}",
                     fontsize=15, fontweight="bold")
        ax.axis("off")
        # also save individual
        wc.to_file(os.path.join(FIG, f"wordcloud_{e}.png"))
    plt.suptitle("Most Frequent Words per Emotion", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "wordcloud_grid.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Generating figures...")
    df = load()
    fig1_distribution(df); print("  class_distribution.png")
    fig2_length(df); print("  text_length.png")
    fig3_model_comparison(); print("  model_comparison.png")
    fig4_confusion(); print("  confusion_matrix.png")
    fig5_wordclouds(df); print("  word clouds")
    print("Done ->", FIG)
