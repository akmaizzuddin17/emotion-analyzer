# 🌈 Social Media Emotion Analyzer

**SAIA 2163 — Final Project · Theme 5**

An interactive Natural Language Processing application that reads a short,
social-media-style message and predicts the **emotion** behind it — joy,
sadness, love, anger, fear, or surprise — with live confidence scores,
data insights, and full model analytics, wrapped in a modern web interface.

> **90.0% accuracy / 0.90 F1** on 2,000 held-out test messages
> (Logistic Regression + TF-IDF).

---

## ✨ What's inside

| Component | File / Folder | Requirement covered |
|-----------|---------------|---------------------|
| Interactive web app | `app.py` | Streamlit application |
| Model development | `notebooks/emotion_analysis.ipynb` | Jupyter notebook |
| Training pipeline | `train_models.py` | Preprocessing, 2 features, 3 models |
| Shared preprocessing | `preprocess.py` | Consistent cleaning everywhere |
| Visualizations | `make_visuals.py`, `reports/figures/` | 5+ charts |
| Technical report | `reports/Technical_Report.docx` / `.pdf` | 8–10 page report |
| Showcase poster | `reports/Poster_A1.pdf` | A1 poster |
| Dataset | `data/*.csv` | 1,000+ labelled samples (18k+2k) |
| Saved models | `models/` | Trained artifacts |

---

## 🚀 Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (First time only) download NLTK data

This happens automatically on first run, but you can pre-fetch it:

```bash
python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4']]"
```

### 3. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

The pre-trained models are already included in `models/`, so the app runs
immediately — no training required.

---

## 🔁 Reproduce the models from scratch (optional)

```bash
python train_models.py     # trains & saves models + metrics  (~30s)
python make_visuals.py     # regenerates all figures
```

Or step through everything interactively in
`notebooks/emotion_analysis.ipynb`.

---

## 🧠 How it works

1. **Preprocessing** (`preprocess.py`) — lowercase, strip URLs/noise,
   tokenize, remove stop-words (**negations kept**), lemmatize.
2. **Feature extraction** — two methods compared:
   - **TF-IDF** (unigram + bigram, 8,000 features)
   - **Word2Vec** (100-dim averaged embeddings)
3. **Models** — Naive Bayes, Logistic Regression, Linear SVM, trained with
   balanced class weights.
4. **Selection** — Logistic Regression + TF-IDF wins (top accuracy **and**
   probability outputs for confidence scores) and powers the app.

| Model | Features | Accuracy | F1 |
|-------|----------|----------|-----|
| Naive Bayes | TF-IDF | 79.1% | 0.76 |
| **Logistic Regression** | **TF-IDF** | **90.0%** | **0.90** |
| Linear SVM | TF-IDF | 90.3% | 0.90 |
| Logistic Regression | Word2Vec | 42.6% | 0.45 |
| Linear SVM | Word2Vec | 48.4% | 0.49 |

---

## 📊 Dataset

The HuggingFace **Emotion** dataset: 20,000 English messages labelled with six
emotions, split into `training.csv` (16k), `validation.csv` (2k), and
`test.csv` (2k). Columns: `text`, `label` (0–5).

Label map: `0 sadness · 1 joy · 2 love · 3 anger · 4 fear · 5 surprise`

---

## 📁 Project structure

```
emotion_project/
├── app.py                     # Streamlit web application
├── preprocess.py              # shared text-cleaning pipeline
├── train_models.py            # train + evaluate + save models
├── make_visuals.py            # generate report/poster figures
├── requirements.txt
├── README.md
├── .gitignore
├── data/                      # train / validation / test CSVs
├── models/                    # saved .pkl + word2vec + metrics
├── notebooks/
│   └── emotion_analysis.ipynb # full model-development walkthrough
├── assets/                    # optional hero.jpg slot
└── reports/
    ├── Technical_Report.docx
    ├── Technical_Report.pdf
    ├── Poster_A1.pdf
    ├── poster.html
    └── figures/               # all generated charts + word clouds
```

---

## 🎨 Optional: add a hero image

Drop a `hero.jpg` into `assets/` and the app fades it behind the headline
automatically. No code changes needed. (See `assets/README.md`.)

---

## 🛠️ Tech stack

Python · scikit-learn · gensim · NLTK · Streamlit · Plotly · WordCloud

---

*Built for SAIA 2163 — Natural Language Processing.*
