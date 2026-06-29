"""
app.py — Social Media Emotion Analyzer  (SAIA 2163 · Theme 5)
============================================================
An interactive NLP web application built with Streamlit.

This version follows the multi-page "editorial paper" design (ported from the
Lovable UI): a sticky TOP NAVIGATION lets the user jump between focused pages
instead of one long scroll, and detecting an emotion smoothly re-themes the
whole page to that emotion's colour.

Pages (mapped to the SAIA 2163 brief's required sections):
  • Home / About      — project, problem, how-to-use, team
  • Data Insights     — sample data, distribution, message length, word clouds, top words
  • Model Performance — metrics, model comparison, confusion matrix

Analysis lives in a sticky right-docked panel that opens on any page. The panel
has an English / Bahasa Melayu toggle — Malay input is auto-translated to
English first (multi-language support), then classified.

Run:  streamlit run app.py
"""

import os
import io
import json
import base64

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud

from preprocess import clean_text, LABEL_MAP

# --------------------------------------------------------------------------- #
# Page config                                                                 #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Emotion Analyzer · SAIA 2163",
    page_icon="🖋️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
DATA = os.path.join(HERE, "data")
ASSETS = os.path.join(HERE, "assets")
ORDER = ["joy", "sadness", "love", "anger", "fear", "surprise"]
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def img_uri(filename: str):
    """Return a base64 data URI for an image in assets/, or None if it's missing.

    Lets us embed local images directly in custom HTML (Streamlit can't serve
    local file paths inside st.markdown). Images are optional — if the file
    isn't there, the UI simply skips it, so nothing ever breaks.
    """
    path = os.path.join(ASSETS, filename)
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(filename)[1].lstrip(".").lower()
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        return "data:image/%s;base64,%s" % (mime, base64.b64encode(f.read()).decode())

# --------------------------------------------------------------------------- #
# Editorial palette — unchanged from the approved UI. Each emotion carries a   #
# soft page "tint", an "accent", and a "deep" ink.                            #
# --------------------------------------------------------------------------- #
NEUTRAL = {"tint": "#FBFAF7", "accent": "#6f6c66", "deep": "#2b2b2b", "label": "Feeling"}
THEME = {
    "joy":      {"tint": "#EAF3DE", "accent": "#5E8C1E", "deep": "#3B5A0E", "label": "Joy"},
    "sadness":  {"tint": "#E6F1FB", "accent": "#2F7BC4", "deep": "#0C447C", "label": "Sadness"},
    "love":     {"tint": "#FBEAF0", "accent": "#C9456F", "deep": "#7A2342", "label": "Love"},
    "anger":    {"tint": "#FAECE7", "accent": "#C9501F", "deep": "#7A2E12", "label": "Anger"},
    "fear":     {"tint": "#E1F5EE", "accent": "#168A66", "deep": "#0B4F3D", "label": "Fear"},
    "surprise": {"tint": "#EEEDFE", "accent": "#6A5FC9", "deep": "#3C3489", "label": "Surprise"},
}

# Pages shown in the top navigation.
PAGES = ["Home", "Data Insights", "Model Performance"]


def face(emotion: str, color: str) -> str:
    """Return an inline hand-drawn SVG face for the given emotion."""
    f = {
        "joy": (
            '<circle cx="50" cy="50" r="38" fill="%s" opacity=".5"/>'
            '<path d="M34 56 Q50 74 66 56" fill="none" stroke="#2b2b2b" stroke-width="3.5" stroke-linecap="round"/>'
            '<circle cx="39" cy="43" r="3.4" fill="#2b2b2b"/><circle cx="61" cy="43" r="3.4" fill="#2b2b2b"/>' % color
        ),
        "sadness": (
            '<circle cx="50" cy="50" r="38" fill="%s" opacity=".5"/>'
            '<path d="M36 64 Q50 52 64 64" fill="none" stroke="#2b2b2b" stroke-width="3.5" stroke-linecap="round"/>'
            '<circle cx="39" cy="44" r="3.4" fill="#2b2b2b"/><circle cx="61" cy="44" r="3.4" fill="#2b2b2b"/>'
            '<path d="M39 50 q-2 7 1 10" fill="none" stroke="#2b2b2b" stroke-width="2.4" stroke-linecap="round"/>' % color
        ),
        "love": (
            '<circle cx="50" cy="50" r="38" fill="%s" opacity=".5"/>'
            '<path d="M36 56 Q50 70 64 56" fill="none" stroke="#2b2b2b" stroke-width="3.5" stroke-linecap="round"/>'
            '<path d="M33 44 q6 -6 12 0" fill="none" stroke="#2b2b2b" stroke-width="3" stroke-linecap="round"/>'
            '<path d="M55 44 q6 -6 12 0" fill="none" stroke="#2b2b2b" stroke-width="3" stroke-linecap="round"/>' % color
        ),
        "anger": (
            '<circle cx="50" cy="50" r="38" fill="%s" opacity=".5"/>'
            '<path d="M36 62 Q50 54 64 62" fill="none" stroke="#2b2b2b" stroke-width="3.5" stroke-linecap="round"/>'
            '<circle cx="39" cy="46" r="3.4" fill="#2b2b2b"/><circle cx="61" cy="46" r="3.4" fill="#2b2b2b"/>'
            '<path d="M32 38 L46 43" stroke="#2b2b2b" stroke-width="3" stroke-linecap="round"/>'
            '<path d="M68 38 L54 43" stroke="#2b2b2b" stroke-width="3" stroke-linecap="round"/>' % color
        ),
        "fear": (
            '<circle cx="50" cy="50" r="38" fill="%s" opacity=".5"/>'
            '<path d="M36 60 q5 -6 9 0 q5 6 9 0 q5 -6 9 0" fill="none" stroke="#2b2b2b" stroke-width="3" stroke-linecap="round"/>'
            '<circle cx="39" cy="44" r="4.6" fill="none" stroke="#2b2b2b" stroke-width="2.6"/>'
            '<circle cx="61" cy="44" r="4.6" fill="none" stroke="#2b2b2b" stroke-width="2.6"/>' % color
        ),
        "surprise": (
            '<circle cx="50" cy="50" r="38" fill="%s" opacity=".5"/>'
            '<ellipse cx="50" cy="61" rx="7" ry="9" fill="none" stroke="#2b2b2b" stroke-width="3"/>'
            '<circle cx="39" cy="43" r="4" fill="#2b2b2b"/><circle cx="61" cy="43" r="4" fill="#2b2b2b"/>'
            '<path d="M31 35 q8 -5 14 -1" fill="none" stroke="#2b2b2b" stroke-width="2.6" stroke-linecap="round"/>'
            '<path d="M55 34 q6 -4 14 1" fill="none" stroke="#2b2b2b" stroke-width="2.6" stroke-linecap="round"/>' % color
        ),
    }
    return (
        '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="%s face">%s</svg>' % (emotion, f[emotion])
    )


# --------------------------------------------------------------------------- #
# Custom CSS — the editorial "paper" identity, themed to the active emotion.   #
# --------------------------------------------------------------------------- #
def inject_css(active="neutral"):
    t = NEUTRAL if active == "neutral" else THEME[active]
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,500;0,9..144,600;1,9..144,400;1,9..144,500&display=swap');

        :root{
          --paper:#FBFAF7; --ink:#2b2b2b; --soft:#6f6c66; --line:rgba(0,0,0,.10);
          --accent:%(accent)s; --tint:%(tint)s; --deep:%(deep)s;
          --body:'DM Sans',-apple-system,'Segoe UI',sans-serif;
        }

        /* kill default streamlit chrome */
        #MainMenu, header, footer {visibility:hidden;}
        .stApp{
          background:var(--tint);
          color:var(--ink);
          transition:background .8s ease;
        }
        .block-container{padding-top:1.4rem; padding-bottom:4rem; margin:0 auto;
          max-width:860px; transition:max-width .35s ease;}
        html, body, [class*="css"]{font-family:var(--body);}

        .es{font-family:'Fraunces','Palatino Linotype','Book Antiqua',Palatino,Georgia,serif;}

        /* ---------- top navigation ---------- */
        .nav-brand{display:flex;align-items:baseline;gap:9px;margin:0 0 2px;}
        .nav-brand .nm{font-family:'Fraunces',Georgia,serif;font-style:italic;font-size:20px;color:var(--ink);}
        .nav-brand .bd{font-size:11px;letter-spacing:.26em;text-transform:uppercase;color:var(--accent);font-weight:500;transition:color .8s;}
        .nav-rule{height:1px;background:var(--line);margin:8px 0 18px;}

        /* the radio that acts as the nav bar */
        div[role="radiogroup"]{flex-direction:row!important;gap:26px!important;flex-wrap:wrap;}
        div[role="radiogroup"] label{margin:0!important;}
        div[role="radiogroup"] label>div:first-child{display:none!important;}  /* hide the dot */
        div[role="radiogroup"] label p{
          font-family:var(--body)!important;font-size:14px!important;color:var(--soft)!important;
          font-weight:500!important;letter-spacing:.01em;padding-bottom:3px;border-bottom:2px solid transparent;
          transition:color .25s ease,border-color .25s ease;}
        div[role="radiogroup"] label:hover p{color:var(--ink)!important;}

        /* ---------- hero ---------- */
        .eb{font-size:12px;letter-spacing:.32em;text-transform:uppercase;color:var(--accent);
          font-weight:500;transition:color .8s;}
        .eh{font-size:44px;line-height:1.06;margin:12px 0 0;font-weight:500;letter-spacing:-.01em;color:var(--ink);}
        .eh .em{font-style:italic;color:var(--accent);transition:color .8s;}
        .el{color:var(--soft);font-size:15.5px;line-height:1.6;max-width:600px;margin:16px 0 0;font-weight:300;}

        /* ---------- result block ---------- */
        .verdict{display:flex;align-items:center;gap:22px;margin:8px 0 4px;}
        .verdict .face{width:104px;height:104px;flex:none;}
        .verdict .emo{font-size:46px;line-height:1;font-weight:500;color:var(--accent);transition:color .8s;}
        .verdict .conf{color:var(--soft);font-size:14.5px;margin-top:6px;}

        .spectrum{display:flex;height:13px;border-radius:100px;overflow:hidden;margin:18px 0 6px;
          border:1px solid var(--line);}
        .spectrum span{transition:width .9s cubic-bezier(.22,1,.36,1);}

        .bars{margin-top:16px;display:flex;flex-direction:column;gap:9px;}
        .bar{display:flex;align-items:center;gap:12px;}
        .bar .nm{width:88px;font-size:13.5px;color:var(--soft);text-transform:capitalize;}
        .tr{flex:1;height:7px;background:rgba(0,0,0,.06);border-radius:100px;overflow:hidden;}
        .fl{height:100%%;border-radius:100px;}
        .pt{width:42px;text-align:right;font-size:12.5px;color:var(--soft);font-variant-numeric:tabular-nums;}

        /* ---------- influential words ---------- */
        .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;}
        .chip{font-size:13px;padding:5px 12px;border-radius:100px;border:1px solid var(--line);
          background:var(--paper);color:var(--deep);transition:color .8s;}
        .chip b{color:var(--accent);font-weight:600;transition:color .8s;}

        /* ---------- section labels ---------- */
        .st-lab{font-size:12.5px;letter-spacing:.24em;text-transform:uppercase;color:var(--soft);
          font-weight:500;margin:30px 0 6px;}
        .home-team-lab{margin-top:10px;}
        .st-sub{color:var(--soft);font-size:14.5px;font-weight:300;margin-bottom:10px;line-height:1.55;}

        /* ---------- six-emotion legend ---------- */
        .legend{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-top:6px;}
        .lg{text-align:center;padding:10px 4px;border-radius:12px;border:1px solid var(--line);background:var(--paper);}
        .lg svg{width:54px;height:54px;}
        .lg div{font-size:12px;color:var(--soft);margin-top:6px;}

        /* ---------- facts ---------- */
        .facts{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:6px;}
        .fct{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:16px 10px;text-align:center;}
        .fct .v{font-size:26px;color:var(--deep);font-weight:500;transition:color .8s;}
        .fct .k{font-size:11.5px;color:var(--soft);margin-top:3px;}

        /* ---------- prose / about ---------- */
        .prose{color:var(--ink);font-size:15px;line-height:1.7;font-weight:300;}
        .prose b{font-weight:500;}
        .team-line{color:var(--soft);font-size:13.5px;line-height:1.7;margin-top:4px;}
        .team-line b{color:var(--ink);font-weight:600;}

        /* ---------- home hero: title overlaid on the watercolor banner ---------- */
        .home-hero{border-radius:18px;margin:0 0 16px;padding:26px 30px;
          border:1px solid var(--line);background:var(--paper);}
        .home-hero.has-bg{
          background-size:cover;background-position:center;
          box-shadow:0 10px 34px rgba(0,0,0,.10);border:none;}
        /* translucent paper scrim so the dark text stays readable on the colours */
        .home-hero.has-bg .home-hero-in{
          background:rgba(251,250,247,.74);
          border-radius:12px;padding:18px 20px;backdrop-filter:blur(2px);}
        .home-hero .eb{margin-bottom:6px;}
        .home-hero .eh{font-size:34px;margin:0;}
        .home-hero .el{margin-top:10px;font-size:14px;line-height:1.55;}

        /* compact one-line "how to use" */
        .howto{color:var(--soft);font-size:13px;line-height:1.7;margin:14px 0 6px;}
        .howto b{color:var(--ink);font-weight:600;}

        /* emotion faces decorative strip (transparent line-art) */
        .faces-strip{display:block;width:100%%;max-width:560px;height:auto;
          margin:6px auto 8px;opacity:.9;mix-blend-mode:multiply;}
        /* ---------- optional social icons in footer ---------- */
        .socials{display:flex;align-items:center;gap:12px;}
        .socials img{width:20px;height:20px;opacity:.55;transition:opacity .2s;
          border-radius:5px;filter:grayscale(20%%);}
        .socials img:hover{opacity:1;}
        /* ---------- optional speech-bubble art (assets/speech.png) ---------- */
        .speech-img{display:block;width:96px;height:auto;margin:0 0 6px;opacity:.85;}

        /* ---------- right-docked analyzer panel ---------- */
        /* the panel column = the last column of the block that holds .panel-anchor */
        [data-testid="stHorizontalBlock"]:has(.panel-anchor) > [data-testid="column"]:last-child{
          position:sticky; top:18px; align-self:flex-start;
          background:var(--paper); border:1px solid var(--line); border-radius:18px;
          padding:14px 16px 16px; box-shadow:0 14px 38px rgba(0,0,0,.10);
        }
        .panel-anchor{display:none;}
        .panel-head{display:flex;align-items:center;justify-content:space-between;margin:2px 0 4px;}
        .panel-title{font-family:'Fraunces',Georgia,serif;font-size:19px;color:var(--ink);font-weight:600;
          display:flex;align-items:center;gap:8px;}
        .panel-icon{width:24px;height:24px;mix-blend-mode:multiply;}
        .panel-hint{color:var(--soft);font-size:12.5px;line-height:1.5;margin:0 0 8px;}
        .panel-open-wrap{display:flex;justify-content:flex-end;margin:-6px 0 2px;}

        /* ---------- streamlit inputs, themed ---------- */
        .stTextArea textarea{
          background:#fff!important;color:var(--ink)!important;
          border:1px solid var(--line)!important;border-radius:11px!important;
          font-size:15px!important;font-family:var(--body)!important;padding:14px 15px!important;}
        .stTextArea textarea:focus{border-color:var(--accent)!important;box-shadow:none!important;}
        .stButton button{
          background:var(--accent)!important;color:#fff!important;border:none!important;
          border-radius:10px!important;font-family:var(--body)!important;font-weight:600!important;
          letter-spacing:.02em!important;padding:11px 24px!important;font-size:14px!important;
          transition:transform .12s ease, background .8s ease!important;text-shadow:0 1px 2px rgba(0,0,0,.1)!important;}
        .stButton button:hover{transform:translateY(-1px)!important;}
        .stSelectbox div[data-baseweb="select"]>div{
          background:#fff!important;border-color:var(--line)!important;border-radius:10px!important;
          font-family:var(--body)!important;color:var(--ink)!important;}
        .stSelectbox div[data-baseweb="select"]>div>div{color:var(--ink)!important;}
        [data-baseweb="select"] span{color:var(--ink)!important;}

        /* sub-tabs */
        .stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);flex-wrap:wrap;}
        .stTabs [data-baseweb="tab"]{font-family:var(--body);font-weight:500;color:var(--soft);
          background:transparent;padding:7px 16px!important;margin:0 2px 2px 0!important;
          border-radius:9px 9px 0 0;transition:color .25s ease,background .25s ease;}
        .stTabs [data-baseweb="tab"]:hover{color:var(--ink);background:rgba(0,0,0,.035);}
        .stTabs [aria-selected="true"]{color:var(--accent)!important;background:rgba(0,0,0,.04)!important;}
        /* animated underline highlight under the active tab */
        .stTabs [data-baseweb="tab-highlight"]{background:var(--accent)!important;height:3px!important;
          border-radius:3px;transition:all .35s cubic-bezier(.22,1,.36,1)!important;}
        /* fade + slide-up the panel contents whenever a tab changes */
        .stTabs [data-baseweb="tab-panel"]{animation:fadeSlide .45s cubic-bezier(.22,1,.36,1);}

        @keyframes fadeSlide{
          from{opacity:0;transform:translateY(10px);}
          to{opacity:1;transform:translateY(0);}
        }
        @keyframes fadeIn{from{opacity:0;}to{opacity:1;}}
        /* gentle page-level fade-in on every rerun / page switch */
        .block-container{animation:fadeIn .5s ease;}

        .stDataFrame{border:1px solid var(--line)!important;border-radius:10px!important;}

        /* metrics (Messages / Emotions / Avg words) — force dark, readable text */
        [data-testid="stMetricValue"]{color:var(--ink)!important;font-weight:600!important;}
        [data-testid="stMetricLabel"]{color:var(--soft)!important;}
        [data-testid="stMetricLabel"] p{color:var(--soft)!important;}

        .foot{margin-top:46px;padding-top:18px;border-top:1px solid var(--line);
          font-size:12px;color:var(--soft);letter-spacing:.04em;
          display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;}
        </style>
        """ % {"accent": t["accent"], "tint": t["tint"], "deep": t["deep"]},
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Cached loaders                                                              #
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS, "best_model.pkl"))
    tfidf = joblib.load(os.path.join(MODELS, "tfidf_vectorizer.pkl"))
    return model, tfidf


@st.cache_data
def load_results():
    with open(os.path.join(MODELS, "all_results.json")) as f:
        results = json.load(f)
    with open(os.path.join(MODELS, "confusion_matrix.json")) as f:
        cm = json.load(f)
    with open(os.path.join(MODELS, "app_metrics.json")) as f:
        app_m = json.load(f)
    return results, cm, app_m


@st.cache_data
def load_dataset():
    df = pd.read_csv(os.path.join(DATA, "training.csv"))
    df["emotion"] = df["label"].map(LABEL_MAP)
    return df


# --------------------------------------------------------------------------- #
# Prediction + explainability                                                 #
# --------------------------------------------------------------------------- #
def predict(text, model, tfidf):
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    probs = model.predict_proba(X)[0]
    order = [LABEL_MAP[i] for i in model.classes_]
    pairs = sorted(zip(order, probs), key=lambda x: -x[1])
    return pairs, cleaned, X


def influential_words(top_emotion, model, tfidf, X, k=6):
    """Words in the input that pushed the model toward the predicted emotion.

    For Logistic Regression, a feature's contribution to a class is
    (tf-idf value) × (class coefficient). We surface the top positive ones.
    """
    try:
        class_idx = list(model.classes_).index(LABEL_MAP_INV[top_emotion])
        coefs = model.coef_[class_idx]
        feats = tfidf.get_feature_names_out()
        x = X.toarray()[0]
        nz = np.nonzero(x)[0]
        contrib = [(feats[i], x[i] * coefs[i]) for i in nz]
        contrib = [c for c in contrib if c[1] > 0]
        contrib.sort(key=lambda c: -c[1])
        return contrib[:k]
    except Exception:
        return []


def _light_layout(fig, height):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a1a1a", family="DM Sans, sans-serif", size=12),
        xaxis=dict(tickfont=dict(color="#1a1a1a", size=11)),
        yaxis=dict(tickfont=dict(color="#1a1a1a", size=11)),
        margin=dict(t=20, b=10),
    )
    return fig


# --------------------------------------------------------------------------- #
# Shared UI pieces                                                            #
# --------------------------------------------------------------------------- #
def top_nav(active):
    """Top navigation built from a horizontal radio styled as nav links."""
    st.markdown(
        '<div class="nav-brand"><span class="nm">Emotion Analyzer</span>'
        '<span class="bd">SAIA 2163</span></div>',
        unsafe_allow_html=True,
    )
    current = st.session_state.get("page", "Home")
    idx = PAGES.index(current) if current in PAGES else 0
    choice = st.radio(
        "nav", PAGES, index=idx, horizontal=True, label_visibility="collapsed",
        key="nav_radio",
    )
    st.session_state["page"] = choice
    st.markdown('<div class="nav-rule"></div>', unsafe_allow_html=True)
    return choice


def legend():
    st.markdown('<div class="st-lab">The six emotions</div>', unsafe_allow_html=True)
    cards = "".join(
        '<div class="lg">%s<div>%s</div></div>' % (face(e, THEME[e]["accent"]), THEME[e]["label"])
        for e in ORDER
    )
    st.markdown('<div class="legend">%s</div>' % cards, unsafe_allow_html=True)


def render_result(pairs, cleaned, infl=None):
    top_e, top_p = pairs[0]
    t = THEME[top_e]
    sure = "very sure" if top_p > 0.7 else "fairly sure" if top_p > 0.45 else "leaning this way"

    st.markdown(
        """
        <div class="verdict">
          <div class="face">%s</div>
          <div>
            <div class="emo es">%s</div>
            <div class="conf">%.1f%%  ·  the message reads as %s</div>
          </div>
        </div>
        """ % (face(top_e, t["accent"]), t["label"], top_p * 100, sure),
        unsafe_allow_html=True,
    )

    segs = "".join(
        '<span style="width:%.1f%%;background:%s;"></span>' % (p * 100, THEME[e]["accent"])
        for e, p in pairs
    )
    st.markdown('<div class="spectrum">%s</div>' % segs, unsafe_allow_html=True)

    rows = "".join(
        '<div class="bar"><div class="nm">%s</div>'
        '<div class="tr"><div class="fl" style="width:%.1f%%;background:%s"></div></div>'
        '<div class="pt">%d%%</div></div>'
        % (THEME[e]["label"], p * 100, THEME[e]["accent"], round(p * 100))
        for e, p in pairs
    )
    st.markdown('<div class="bars">%s</div>' % rows, unsafe_allow_html=True)

    if infl:
        st.markdown('<div class="st-lab">Words that influenced this</div>', unsafe_allow_html=True)
        chips = "".join('<span class="chip"><b>%s</b></span>' % w for w, _ in infl)
        st.markdown('<div class="chips">%s</div>' % chips, unsafe_allow_html=True)

    if cleaned:
        st.caption("Model input after preprocessing →  %s" % cleaned)


# --------------------------------------------------------------------------- #
# PAGE: Home / About                                                          #
# --------------------------------------------------------------------------- #
def page_home(active, app_m):
    word = "feeling" if active == "neutral" else THEME[active]["label"].lower()

    # --- Hero: title + description sit ON the watercolor banner (one compact block).
    # This combines the project title, description and problem statement, and saves
    # vertical space so the whole Home page fits without scrolling.
    hero = img_uri("hero.jpg") or img_uri("hero.png")
    hero_bg = "background-image:url(%s);" % hero if hero else ""
    st.markdown(
        """
        <div class="home-hero %(cls)s" style="%(bg)s">
          <div class="home-hero-in">
            <div class="eb">SAIA 2163 · Theme 5 · Emotion Analyzer</div>
            <h1 class="eh es">Read the <span class="em">%(word)s</span> behind the words.</h1>
            <p class="el">An NLP app that reads a social-media message and predicts the emotion
            underneath joy, sadness, love, anger, fear, or surprise with live confidence
            scores. Feeling is hard to measure at scale; a Logistic Regression + TF-IDF model
            trained on 18,000 real messages does it instantly.</p>
          </div>
        </div>
        """ % {"cls": "has-bg" if hero else "", "bg": hero_bg, "word": word},
        unsafe_allow_html=True,
    )

    facts(app_m)

    # How to use — compact single line.
    st.markdown(
        '<p class="howto"><b>How to use</b> &nbsp;·&nbsp; '
        '<b>1</b> Click <b>✎ Analyze text</b> (top-right) to open the panel &nbsp;→&nbsp; '
        '<b>2</b> choose <b>English</b> or <b>Bahasa Melayu</b>, type a message &nbsp;→&nbsp; '
        '<b>3</b> read the emotion, confidence &amp; influential words &nbsp;→&nbsp; '
        '<b>4</b> explore <b>Data Insights</b> &amp; <b>Model Performance</b>.</p>',
        unsafe_allow_html=True,
    )

    # Emotion faces — slim decorative strip (transparent line-art), replaces the
    # bulkier SVG legend while still signalling the emotion theme.
    faces = img_uri("faces.png")
    if faces:
        st.markdown('<img class="faces-strip" src="%s" alt="emotion faces"/>' % faces,
                    unsafe_allow_html=True)
    else:
        legend()

    st.markdown('<div class="st-lab home-team-lab">Team</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="team-line">'
        '<b>Adam</b> · Data &amp; preprocessing&nbsp;&nbsp;•&nbsp;&nbsp;'
        '<b>Harraz</b> · Modelling &amp; evaluation&nbsp;&nbsp;•&nbsp;&nbsp;'
        '<b>Akma</b> · Streamlit app &amp; UI&nbsp;&nbsp;•&nbsp;&nbsp;'
        '<b>Aryl</b> · Report &amp; visualizations'
        '</p>',
        unsafe_allow_html=True,
    )


def facts(app_m):
    st.markdown('<div class="st-lab">Under the hood</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="facts">
          <div class="fct"><div class="v">%.0f%%</div><div class="k">Test accuracy</div></div>
          <div class="fct"><div class="v">18K</div><div class="k">Training messages</div></div>
          <div class="fct"><div class="v">6</div><div class="k">Emotions</div></div>
          <div class="fct"><div class="v">TF-IDF</div><div class="k">+ Logistic Reg.</div></div>
        </div>
        """ % (app_m["accuracy"] * 100,),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# PAGE: Data Insights                                                         #
# --------------------------------------------------------------------------- #
def page_insights(df):
    st.markdown('<div class="st-lab">Inside the data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="st-sub">What 18,000 labelled messages look like before any model sees them.</div>',
        unsafe_allow_html=True,
    )

    t1, t2, t3, t4, t5 = st.tabs(
        ["Sample data", "Emotion mix", "Message length", "Top words", "Word clouds"]
    )

    with t1:
        st.markdown('<div class="st-sub">A random peek at the labelled dataset.</div>',
                    unsafe_allow_html=True)
        sample = df.sample(8, random_state=7)[["text", "emotion"]].reset_index(drop=True)
        st.dataframe(sample, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Messages", f"{len(df):,}")
        c2.metric("Emotions", df["emotion"].nunique())
        c3.metric("Avg words", f"{df['text'].str.split().str.len().mean():.1f}")

    with t2:
        counts = df["emotion"].value_counts().reindex(ORDER)
        fig = go.Figure(go.Bar(
            x=[THEME[e]["label"] for e in counts.index], y=counts.values,
            marker=dict(color=[THEME[e]["accent"] for e in counts.index]),
            text=counts.values, textposition="outside",
            textfont=dict(color="#1a1a1a", size=12),
        ))
        _light_layout(fig, 400)
        fig.update_layout(xaxis=dict(showgrid=False, tickfont=dict(color="#1a1a1a")),
                          yaxis=dict(showgrid=False, visible=False))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The dataset is imbalanced — joy and sadness dominate, surprise is rarest.")

    with t3:
        d = df.copy()
        d["wlen"] = d["text"].str.split().str.len()
        fig = go.Figure()
        for e in ORDER:
            fig.add_trace(go.Violin(
                y=d[d.emotion == e]["wlen"], name=THEME[e]["label"],
                line_color=THEME[e]["accent"], box_visible=True, meanline_visible=True,
            ))
        _light_layout(fig, 430)
        fig.update_layout(
            yaxis=dict(title=dict(text="Words per message", font=dict(color="#1a1a1a", size=13)),
                       range=[0, 60], gridcolor="rgba(0,0,0,0.06)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        pick = st.selectbox(
            "Emotion ", ORDER, format_func=lambda e: THEME[e]["label"], key="topwords_pick",
        )
        sub = df[df.emotion == pick]["text"].apply(clean_text)
        words = " ".join(sub).split()
        top = pd.Series(words).value_counts().head(20)[::-1]
        fig = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation="h",
            marker=dict(color=THEME[pick]["accent"]),
        ))
        _light_layout(fig, 480)
        fig.update_layout(
            xaxis=dict(title=dict(text="Count", font=dict(color="#1a1a1a")), gridcolor="rgba(0,0,0,0.06)"),
            margin=dict(t=20, b=10, l=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Top 20 most frequent words for the selected emotion (after preprocessing).")

    with t5:
        pick = st.selectbox(
            "Emotion", ORDER, format_func=lambda e: THEME[e]["label"], key="wc_pick",
        )
        sub = df[df.emotion == pick]["text"].apply(clean_text)
        wc = WordCloud(
            width=1000, height=400, background_color="#FBFAF7",
            colormap="copper", max_words=70,
        ).generate(" ".join(sub))
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        st.image(buf.getvalue(), use_container_width=True)


# --------------------------------------------------------------------------- #
# PAGE: Model Performance                                                     #
# --------------------------------------------------------------------------- #
def page_performance(results, cm, app_m):
    st.markdown('<div class="st-lab">How well it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="st-sub">Measured on 2,000 held-out messages the model never saw during training.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="facts">
          <div class="fct"><div class="v">%.1f%%</div><div class="k">Accuracy</div></div>
          <div class="fct"><div class="v">%.1f%%</div><div class="k">F1 (weighted)</div></div>
          <div class="fct"><div class="v">%.1f%%</div><div class="k">Precision</div></div>
          <div class="fct"><div class="v">%.1f%%</div><div class="k">Recall</div></div>
        </div>
        """ % (app_m["accuracy"] * 100, app_m["f1"] * 100,
               app_m["precision"] * 100, app_m["recall"] * 100),
        unsafe_allow_html=True,
    )
    st.write("")

    t1, t2, t3 = st.tabs(["Model comparison", "Confusion matrix", "Model info"])
    with t1:
        labels = [f"{r['model']} · {r['feature']}" for r in results]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Accuracy", x=labels, y=[r["accuracy"] for r in results],
                             marker_color="#2F7BC4"))
        fig.add_trace(go.Bar(name="F1", x=labels, y=[r["f1"] for r in results],
                             marker_color="#5E8C1E"))
        _light_layout(fig, 430)
        fig.update_layout(
            barmode="group", margin=dict(t=20, b=80),
            yaxis=dict(range=[0, 1], gridcolor="rgba(0,0,0,0.06)"),
            xaxis=dict(tickangle=-20),
            legend=dict(orientation="h", y=1.12, font=dict(color="#1a1a1a", size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "TF-IDF clearly outperforms averaged Word2Vec here — short messages give "
            "Word2Vec too few words to average reliably, while TF-IDF captures the strong "
            "emotion keywords directly."
        )

    with t2:
        mat = np.array(cm["matrix"])
        norm = mat / mat.sum(axis=1, keepdims=True)
        fig = px.imshow(
            norm, x=[THEME[e]["label"] for e in cm["labels"]],
            y=[THEME[e]["label"] for e in cm["labels"]],
            color_continuous_scale="YlGn",
            labels=dict(x="Predicted", y="Actual", color="Proportion"), text_auto=False,
        )
        fig.update_traces(text=mat, texttemplate="%{text}", textfont=dict(color="#1a1a1a", size=13))
        _light_layout(fig, 470)
        fig.update_layout(
            xaxis=dict(tickfont=dict(color="#1a1a1a", size=12)),
            yaxis=dict(tickfont=dict(color="#1a1a1a", size=12)),
            coloraxis_colorbar=dict(tickfont=dict(color="#1a1a1a")),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rows are the true emotion, columns the prediction. A strong diagonal means good accuracy.")

    with t3:
        st.markdown(
            '<p class="prose">The deployed model is <b>Logistic Regression</b> on <b>TF-IDF</b> features '
            '(unigrams + bigrams, 8,000 features). It was chosen over Naive Bayes and Linear SVM because it '
            'matches the top accuracy <i>and</i> gives calibrated probability outputs, which power the '
            'confidence scores on the Analyze page.</p>',
            unsafe_allow_html=True,
        )
        rows = "".join(
            "<tr><td style='padding:8px 10px;border-bottom:1px solid var(--line)'>%s · %s</td>"
            "<td style='padding:8px 10px;border-bottom:1px solid var(--line);text-align:right'>%.1f%%</td>"
            "<td style='padding:8px 10px;border-bottom:1px solid var(--line);text-align:right'>%.2f</td></tr>"
            % (r["model"], r["feature"], r["accuracy"] * 100, r["f1"]) for r in results
        )
        st.markdown(
            "<table style='width:100%%;border-collapse:collapse;font-size:14px;color:var(--ink)'>"
            "<tr><th style='text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)'>Model</th>"
            "<th style='text-align:right;padding:8px 10px;border-bottom:1px solid var(--line)'>Accuracy</th>"
            "<th style='text-align:right;padding:8px 10px;border-bottom:1px solid var(--line)'>F1</th></tr>"
            "%s</table>" % rows,
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# PAGE: Malay (multi-language support)                                        #
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
@st.cache_data(show_spinner=False)
def translate_ms_to_en(text):
    """Translate Malay → English. Returns (english_text, error_or_None).

    Tries Google Translate first, then falls back to MyMemory if Google is
    blocked or rate-limited, so a single flaky request doesn't break the page.
    Results are cached so re-runs of the same text don't re-hit the network.
    """
    errors = []
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source="ms", target="en").translate(text)
        if out and out.strip():
            return out, None
    except Exception as e:
        errors.append("Google: %s" % e)

    try:
        from deep_translator import MyMemoryTranslator
        out = MyMemoryTranslator(source="ms-MY", target="en-US").translate(text)
        if out and out.strip():
            return out, None
    except Exception as e:
        errors.append("MyMemory: %s" % e)

    return None, " | ".join(errors) if errors else "no translation returned"


# --------------------------------------------------------------------------- #
# Right-docked analyzer panel                                                 #
# --------------------------------------------------------------------------- #
def analyzer_panel(model, tfidf):
    """The sticky right-docked text analyzer (shown when the panel is open)."""
    # Marker that scopes the sticky/card CSS to THIS column only.
    st.markdown('<span class="panel-anchor"></span>', unsafe_allow_html=True)

    head_l, head_r = st.columns([5, 1])
    with head_l:
        speech = img_uri("speech.png") or img_uri("speech.jpg")
        icon = '<img class="panel-icon" src="%s" alt=""/>' % speech if speech else "✎ "
        st.markdown('<div class="panel-title">%sAnalyze</div>' % icon, unsafe_allow_html=True)
    with head_r:
        if st.button("✕", key="close_panel", help="Close panel"):
            st.session_state["panel_open"] = False
            st.rerun()

    # Language toggle at the top — English (default) or Bahasa Melayu.
    lang = st.radio(
        "lang", ["English", "Bahasa Melayu"], horizontal=True,
        label_visibility="collapsed", key="panel_lang",
    )
    is_my = lang.startswith("Bahasa")

    if is_my:
        st.markdown('<p class="panel-hint">Taip mesej dalam Bahasa Melayu — ia diterjemah ke '
                    'Bahasa Inggeris dahulu, kemudian emosinya dianalisis.</p>',
                    unsafe_allow_html=True)
        examples = {
            "Pilih contoh…": "",
            "Gembira": "saya rasa sangat gembira dan bersyukur hari ini",
            "Sedih": "saya rasa sedih dan keseorangan sejak kebelakangan ini",
            "Takut": "saya rasa cemas dan takut tentang keputusan esok",
            "Marah": "saya sangat marah kerana mereka menipu saya lagi",
            "Sayang": "saya sangat menyayangi keluarga saya",
        }
        placeholder = "cth. saya rasa hari ini sangat bermakna…"
        btn_label, ex_label, ta_label = "Analisis emosi  →", "contoh_my", "panel_msg_my"
    else:
        st.markdown('<p class="panel-hint">Type any message — the model reads the emotion '
                    'underneath and the page takes on its colour.</p>', unsafe_allow_html=True)
        examples = {
            "Pick an example…": "",
            "Joyful": "i feel so happy and cheerful today everything is wonderful",
            "Heartfelt": "i feel so grateful and loved being around my family",
            "Anxious": "i feel so nervous and scared about the results tomorrow",
            "Angry": "i am so furious that they lied to me again",
            "Surprised": "i can't believe this just happened it caught me totally off guard",
        }
        placeholder = "e.g. i feel like everything is finally falling into place…"
        btn_label, ex_label, ta_label = "Analyze emotion  →", "examples_en", "panel_msg_en"

    pick = st.selectbox(ex_label, list(examples.keys()), label_visibility="collapsed")
    text = st.text_area(
        ta_label, value=examples[pick], height=120, label_visibility="collapsed",
        placeholder=placeholder,
    )

    if st.button(btn_label, key="panel_go", use_container_width=True):
        if not text.strip():
            st.warning("Taip mesej dahulu. (Type a message first.)" if is_my
                       else "Type a message first, then analyze.")
        else:
            ok, translation, model_input = True, None, text
            if is_my:
                with st.spinner("Menterjemah… (translating)"):
                    english, err = translate_ms_to_en(text)
                if err or not english:
                    st.error("Terjemahan gagal. Sila cuba lagi sebentar. "
                             "(Translation failed — please try again.)")
                    ok = False
                else:
                    translation, model_input = english, english
            if ok:
                pairs, cleaned, X = predict(model_input, model, tfidf)
                infl = influential_words(pairs[0][0], model, tfidf, X)
                st.session_state["active"] = pairs[0][0]
                st.session_state["result"] = {
                    "pairs": pairs, "cleaned": cleaned, "infl": infl,
                    "translation": translation,
                }
                st.rerun()

    res = st.session_state.get("result")
    if res:
        if res.get("translation"):
            st.markdown(
                '<p class="panel-hint" style="margin-top:8px"><b>Translation:</b> %s</p>'
                % res["translation"], unsafe_allow_html=True)
        render_result(res["pairs"], res["cleaned"], res.get("infl"))


def render_page(page, active, app_m, df, results, cm, model, tfidf):
    """Render whichever page is selected (used inside the left/body column)."""
    if page == "Home":
        page_home(active, app_m)
    elif page == "Data Insights":
        page_insights(df)
    elif page == "Model Performance":
        page_performance(results, cm, app_m)

    # Optional social icons — each shows only if its file exists in assets/.
    social_files = [("ig.png", "Instagram"), ("x.png", "X"),
                    ("fb.png", "Facebook"), ("threads.png", "Threads")]
    icons = "".join(
        '<img src="%s" alt="%s" title="%s"/>' % (img_uri(fn), name, name)
        for fn, name in social_files if img_uri(fn)
    )
    socials_html = '<span class="socials">%s</span>' % icons if icons else \
        '<span>SAIA 2163 · Theme 5 · Emotion Analyzer</span>'
    st.markdown(
        '<div class="foot"><span>Type a sentence with feeling and watch the colour shift</span>'
        '%s</div>' % socials_html,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    active = st.session_state.get("active", "neutral")
    panel_open = st.session_state.get("panel_open", False)

    inject_css(active)
    # Widen the page when the dock is open so content shifts left + panel fits.
    st.markdown(
        "<style>.block-container{max-width:%s;}</style>"
        % ("1320px" if panel_open else "860px"),
        unsafe_allow_html=True,
    )

    model, tfidf = load_model()
    results, cm, app_m = load_results()
    df = load_dataset()

    if panel_open:
        body, panel = st.columns([2.3, 1], gap="large")
        with body:
            page = top_nav(active)
            render_page(page, active, app_m, df, results, cm, model, tfidf)
        with panel:
            analyzer_panel(model, tfidf)
    else:
        page = top_nav(active)
        # "Open analyzer" toggle, right-aligned just under the nav.
        st.markdown('<div class="panel-open-wrap">', unsafe_allow_html=True)
        oc1, oc2 = st.columns([7, 2])
        with oc2:
            if st.button("✎  Analyze text", key="open_panel", use_container_width=True):
                st.session_state["panel_open"] = True
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        render_page(page, active, app_m, df, results, cm, model, tfidf)


if __name__ == "__main__":
    main()
# editorial paper UI — multi-page top-nav + Malay support
