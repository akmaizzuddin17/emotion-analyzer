"""
app.py — Social Media Emotion Analyzer
======================================
An interactive NLP web application (SAIA 2163, Theme 5).

The app loads the pre-trained Logistic Regression + TF-IDF model and lets a
user type any social-media-style message to instantly detect the emotion
behind it. The interface follows an editorial "paper" identity: warm light
canvas, Jost + Palatino type, hand-drawn emotion faces, and a page colour
that shifts to match whatever emotion the model reads.

Run:  streamlit run app.py
"""

import os
import json
import io

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud

from preprocess import clean_text, LABEL_MAP, EMOTION_META

# --------------------------------------------------------------------------- #
# Page config                                                                 #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Emotion Analyzer · SAIA 2163",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
DATA = os.path.join(HERE, "data")
EMOTIONS = [LABEL_MAP[i] for i in range(6)]
ORDER = ["joy", "sadness", "love", "anger", "fear", "surprise"]

# --------------------------------------------------------------------------- #
# Editorial palette — ported from the approved web-UI requirement.            #
# Each emotion carries a soft page "tint", an "accent", and a "deep" ink.     #
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
# Custom CSS — the editorial "paper" identity, themed to the active emotion.  #
# --------------------------------------------------------------------------- #
def inject_css(active="neutral"):
    t = NEUTRAL if active == "neutral" else THEME[active]
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Jost:wght@300;400;500;600&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&display=swap');

        :root{
          --paper:#FBFAF7; --ink:#2b2b2b; --soft:#6f6c66; --line:rgba(0,0,0,.10);
          --accent:%(accent)s; --tint:%(tint)s; --deep:%(deep)s;
          --body:'Futura','Futura PT','Century Gothic','Jost',-apple-system,sans-serif;
        }

        /* kill default streamlit chrome */
        #MainMenu, header, footer {visibility:hidden;}
        .stApp{
          background:var(--tint);
          color:var(--ink);
          transition:background .8s ease;
        }
        .block-container{padding-top:2.4rem; padding-bottom:4rem; max-width:760px;}
        html, body, [class*="css"]{font-family:var(--body);}

        .es{font-family:'Playfair Display','Palatino Linotype','Book Antiqua',Palatino,Georgia,serif;}

        /* ---------- hero ---------- */
        .eb{font-size:12px;letter-spacing:.32em;text-transform:uppercase;color:var(--accent);
          font-weight:500;transition:color .8s;}
        .eh{font-size:44px;line-height:1.06;margin:12px 0 0;font-weight:500;letter-spacing:-.01em;color:var(--ink);}
        .eh .em{font-style:italic;color:var(--accent);transition:color .8s;}
        .el{color:var(--soft);font-size:15.5px;line-height:1.6;max-width:560px;margin:16px 0 0;font-weight:300;}

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

        /* ---------- section labels ---------- */
        .st-lab{font-size:12.5px;letter-spacing:.24em;text-transform:uppercase;color:var(--soft);
          font-weight:500;margin:34px 0 6px;}
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

        /* tabs */
        .stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid var(--line);}
        .stTabs [data-baseweb="tab"]{font-family:var(--body);font-weight:500;color:var(--soft);background:transparent;}
        .stTabs [aria-selected="true"]{color:var(--ink)!important;}

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
# Prediction                                                                  #
# --------------------------------------------------------------------------- #
def predict(text, model, tfidf):
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    probs = model.predict_proba(X)[0]
    order = [LABEL_MAP[i] for i in model.classes_]
    pairs = sorted(zip(order, probs), key=lambda x: -x[1])
    return pairs, cleaned


# --------------------------------------------------------------------------- #
# UI sections                                                                 #
# --------------------------------------------------------------------------- #
def hero(active):
    word = NEUTRAL["label"].lower() if active == "neutral" else THEME[active]["label"].lower()
    st.markdown(
        """
        <div class="eb">SAIA 2163 · Emotion Analyzer</div>
        <h1 class="eh es">Read the <span class="em">%s</span><br>behind the words.</h1>
        <p class="el">Type a message the way you would post it. The analyzer reads the emotion
        underneath — and the page takes on its colour. Trained on 18,000 real messages with
        Logistic Regression + TF-IDF.</p>
        """ % word,
        unsafe_allow_html=True,
    )


def analyzer(model, tfidf):
    st.markdown('<div class="st-lab">Try it live</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="st-sub">The model reads the feeling, not just the words. '
        'Pick an example or write your own.</div>',
        unsafe_allow_html=True,
    )

    examples = {
        "Pick an example…": "",
        "Joyful": "i feel so happy and cheerful today everything is wonderful",
        "Heartfelt": "i feel so grateful and loved being around my family",
        "Anxious": "i feel so nervous and scared about the results tomorrow",
        "Angry": "i am so furious that they lied to me again",
        "Surprised": "i can't believe this just happened it caught me totally off guard",
    }

    c1, c2 = st.columns([1, 2])
    with c1:
        pick = st.selectbox("Quick examples", list(examples.keys()), label_visibility="collapsed")
    default = examples[pick]

    text = st.text_area(
        "msg", value=default, height=120, label_visibility="collapsed",
        placeholder="e.g. i feel like everything is finally falling into place…",
    )
    go_btn = st.button("Analyze emotion  →")

    if go_btn:
        if text.strip():
            pairs, cleaned = predict(text, model, tfidf)
            st.session_state["active"] = pairs[0][0]
            st.session_state["result"] = {"pairs": pairs, "cleaned": cleaned}
            # rerun so the whole page re-themes to the detected emotion; the
            # result is rendered below from session_state, so it persists.
            st.rerun()
        else:
            st.warning("Type a message first, then analyze.")

    # Render the most recent result (survives the re-theme rerun).
    res = st.session_state.get("result")
    if res:
        render_result(res["pairs"], res["cleaned"])


def render_result(pairs, cleaned):
    top_e, top_p = pairs[0]
    t = THEME[top_e]
    sure = "very sure" if top_p > 0.7 else "fairly sure" if top_p > 0.45 else "leaning this way"

    # verdict + face
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

    # spectrum
    segs = "".join(
        '<span style="width:%.1f%%;background:%s;"></span>' % (p * 100, THEME[e]["accent"])
        for e, p in pairs
    )
    st.markdown('<div class="spectrum">%s</div>' % segs, unsafe_allow_html=True)

    # probability bars
    rows = "".join(
        '<div class="bar"><div class="nm">%s</div>'
        '<div class="tr"><div class="fl" style="width:%.1f%%;background:%s"></div></div>'
        '<div class="pt">%d%%</div></div>'
        % (THEME[e]["label"], p * 100, THEME[e]["accent"], round(p * 100))
        for e, p in pairs
    )
    st.markdown('<div class="bars">%s</div>' % rows, unsafe_allow_html=True)

    if cleaned:
        st.caption("Model input after preprocessing →  %s" % cleaned)


def legend():
    st.markdown('<div class="st-lab">The six emotions</div>', unsafe_allow_html=True)
    cards = "".join(
        '<div class="lg">%s<div>%s</div></div>' % (face(e, THEME[e]["accent"]), THEME[e]["label"])
        for e in ORDER
    )
    st.markdown('<div class="legend">%s</div>' % cards, unsafe_allow_html=True)


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


def _light_layout(fig, height):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1a1a1a", family="Futura, Century Gothic, sans-serif", size=12),
        xaxis=dict(tickfont=dict(color="#1a1a1a", size=11)),
        yaxis=dict(tickfont=dict(color="#1a1a1a", size=11)),
        margin=dict(t=20, b=10),
    )
    return fig


def insights(df):
    st.markdown('<div class="st-lab">Inside the data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="st-sub">What 18,000 labelled messages look like before any model sees them.</div>',
        unsafe_allow_html=True,
    )

    t1, t2, t3 = st.tabs(["Emotion mix", "Message length", "Word clouds"])

    with t1:
        counts = df["emotion"].value_counts().reindex(ORDER)
        fig = go.Figure(go.Bar(
            x=[THEME[e]["label"] for e in counts.index], y=counts.values,
            marker=dict(color=[THEME[e]["accent"] for e in counts.index]),
            text=counts.values, textposition="outside",
            textfont=dict(color="#1a1a1a", size=12),
        ))
        _light_layout(fig, 400)
        fig.update_layout(xaxis=dict(showgrid=False, tickfont=dict(color="#1a1a1a")), yaxis=dict(showgrid=False, visible=False))
        st.plotly_chart(fig, use_container_width=True)

    with t2:
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
            yaxis=dict(title=dict(text="Words per message", font=dict(color="#1a1a1a", size=13)), range=[0, 60], gridcolor="rgba(0,0,0,0.06)"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        pick = st.selectbox(
            "Emotion", ORDER, format_func=lambda e: THEME[e]["label"],
        )
        sub = df[df.emotion == pick]["text"].apply(clean_text)
        wc = WordCloud(
            width=1000, height=400, background_color="#FBFAF7",
            colormap="copper", max_words=70,
        ).generate(" ".join(sub))
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        st.image(buf.getvalue(), use_container_width=True)


def performance(results, cm, app_m):
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
        """ % (app_m["accuracy"] * 100, app_m["f1"] * 100, app_m["precision"] * 100, app_m["recall"] * 100),
        unsafe_allow_html=True,
    )
    st.write("")

    t1, t2 = st.tabs(["Model comparison", "Confusion matrix"])
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


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    active = st.session_state.get("active", "neutral")
    inject_css(active)

    model, tfidf = load_model()
    results, cm, app_m = load_results()
    df = load_dataset()

    hero(active)
    st.write("")
    analyzer(model, tfidf)
    legend()
    facts(app_m)
    st.write("")
    insights(df)
    st.write("")
    performance(results, cm, app_m)

    st.markdown(
        '<div class="foot"><span>Type a sentence with feeling and watch the colour shift</span>'
        '<span>SAIA 2163 · Theme 5 · Emotion Analyzer</span></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
# editorial paper UI — applied per requirement
