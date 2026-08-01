"""
Fake Job Posting Detector — Streamlit app
Loads the model trained in the notebook (best_fake_job_model.pkl) and gives
an explainable prediction: REAL/FAKE gauge, colour-coded word contributions,
ranked top words, and a flip/robustness analysis.

Run:
    streamlit run app.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.explain import analyze

# ---------------------------------------------------------------------------
# Page config — do this first, before any other st.* call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "models" / "best_fake_job_model.pkl"

EXAMPLES = {
    "Real example": {
        "title": "Senior Data Analyst",
        "description": (
            "We are looking for a Senior Data Analyst to join our growing analytics team. "
            "You will work closely with product and marketing teams to build dashboards, "
            "analyze user behavior, and present insights to leadership. Requirements include "
            "3+ years of experience with SQL, Python, and a BI tool such as Tableau or Power BI. "
            "This is a full-time position based in our downtown office with standard benefits "
            "including health insurance and paid time off."
        ),
    },
    "Fake example": {
        "title": "Work From Home Data Entry - Earn $500/Day",
        "description": (
            "No experience needed! Immediate hiring for data entry position. Work only 2 hours "
            "a day and earn guaranteed cash weekly. Just send your bank account details and a "
            "small processing fee of $50 to get started today. Limited positions available, "
            "apply now, no interview required, click the link below to claim your spot."
        ),
    },
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 10% 0%, #101827 0%, #0b0f19 55%, #060810 100%);
        }
        section[data-testid="stSidebar"] {
            background: #0e1420;
            border-right: 1px solid #1f2937;
        }
        .hero {
            padding: 1.6rem 2rem;
            border-radius: 18px;
            background: linear-gradient(120deg, rgba(37,99,235,0.18), rgba(16,185,129,0.12));
            border: 1px solid rgba(148,163,184,0.15);
            margin-bottom: 1.4rem;
        }
        .hero h1 {
            font-size: 2.1rem;
            margin-bottom: 0.2rem;
            background: linear-gradient(90deg, #60a5fa, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            color: #94a3b8;
            font-size: 1rem;
            margin: 0;
        }
        .metric-card {
            border-radius: 14px;
            padding: 1rem 1.2rem;
            background: #111827;
            border: 1px solid #1f2937;
        }
        .metric-card h3 {
            margin: 0;
            font-size: 0.85rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
        }
        .word-chip {
            display: inline-block;
            padding: 5px 10px;
            margin: 3px;
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.92rem;
            color: #0b0f19;
        }
        .label-badge {
            display: inline-block;
            padding: 0.35rem 1rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1rem;
            letter-spacing: 0.03em;
        }
        .badge-real { background: rgba(52, 211, 153, 0.18); color: #34d399; border: 1px solid #34d399; }
        .badge-fake { background: rgba(248, 113, 113, 0.18); color: #f87171; border: 1px solid #f87171; }
        .contrib-row {
            display: flex; justify-content: space-between;
            padding: 6px 10px; border-radius: 8px; margin-bottom: 4px;
            background: #111827; border: 1px solid #1f2937; font-size: 0.9rem;
        }
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_model(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload  # {"model_name": ..., "model": ..., "vectorizer": ...}


# ---------------------------------------------------------------------------
# Visual components
# ---------------------------------------------------------------------------
def render_gauge(prob_real: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob_real * 100,
            number={"suffix": "% REAL", "font": {"size": 30, "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8", "tickfont": {"color": "#94a3b8"}},
                "bar": {"color": "#0b0f19", "thickness": 0.001},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "#dc2626"},
                    {"range": [25, 50], "color": "#f59e0b"},
                    {"range": [50, 75], "color": "#a3e635"},
                    {"range": [75, 100], "color": "#16a34a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.85,
                    "value": prob_real * 100,
                },
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0"},
    )
    return fig


def render_word_chips(impacts):
    """Colour-coded word chips: green = pushes toward REAL, red = pushes toward FAKE."""
    if not impacts:
        st.info("No tokens to analyze.")
        return
    max_abs = max((abs(imp) for _, imp in impacts), default=1e-6) or 1e-6
    html = ""
    for word, imp in impacts:
        norm = imp / max_abs
        if norm >= 0:
            g = int(120 + 100 * min(norm, 1.0))
            color = f"rgb({255 - g}, {min(255, 140 + g)}, {255 - g})"
        else:
            r = int(120 + 100 * min(abs(norm), 1.0))
            color = f"rgb({min(255, 140 + r)}, {255 - r}, {255 - r})"
        html += f'<span class="word-chip" style="background:{color};">{word}</span>'
    st.markdown(html, unsafe_allow_html=True)


def render_top_words(top_words):
    for word, imp in top_words:
        direction = "REAL" if imp >= 0 else "FAKE"
        color = "#34d399" if imp >= 0 else "#f87171"
        st.markdown(
            f"""
            <div class="contrib-row">
                <span><b>{word}</b></span>
                <span style="color:{color};">{direction} &nbsp; ({imp:+.4f})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main():
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>🕵️ Fake Job Posting Detector</h1>
            <p>Paste a job title and description to get an explainable REAL / FAKE prediction —
            not just a label, but <i>why</i> the model thinks so.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    payload = load_model(MODEL_PATH)

    with st.sidebar:
        st.header("⚙️ Model")
        if payload is None:
            st.error(
                f"Could not find `{MODEL_PATH.name}` in the `models/` folder.\n\n"
                "Download it from your training notebook (Step 14) and place it at:\n"
                f"`{MODEL_PATH}`"
            )
        else:
            st.success(f"Loaded: **{payload.get('model_name', 'unknown model')}**")

        st.markdown("---")
        st.header("📋 Try an example")
        for label, example in EXAMPLES.items():
            if st.button(label, use_container_width=True):
                st.session_state["title_input"] = example["title"]
                st.session_state["desc_input"] = example["description"]

        st.markdown("---")
        st.caption(
            "Green = pushes toward REAL · Red = pushes toward FAKE · darker = stronger effect"
        )

    if payload is None:
        st.stop()

    model = payload["model"]
    vectorizer = payload["vectorizer"]

    col_title, col_desc = st.columns([1, 2])
    with col_title:
        title = st.text_input(
            "Job title",
            key="title_input",
            placeholder="e.g. Senior Data Analyst",
        )
    with col_desc:
        description = st.text_area(
            "Job description",
            key="desc_input",
            height=140,
            placeholder="Paste the full job description here...",
        )

    analyze_clicked = st.button("🔍 Analyze posting", type="primary", use_container_width=True)

    if analyze_clicked:
        if not title.strip() and not description.strip():
            st.warning("Enter a job title or description first.")
            st.stop()

        with st.spinner("Analyzing word-by-word contributions..."):
            result = analyze(title, description, model, vectorizer)

        st.markdown("### Result")

        badge_class = "badge-real" if result["label"] == "REAL" else "badge-fake"
        st.markdown(
            f'<span class="label-badge {badge_class}">Prediction: {result["label"]}</span>',
            unsafe_allow_html=True,
        )
        if result["truncated"]:
            st.caption("⚠️ Input truncated to the first 80 words for speed.")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                f'<div class="metric-card"><h3>REAL probability</h3>'
                f'<div class="value" style="color:#34d399;">{result["baseline_prob_real"]*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><h3>FAKE probability</h3>'
                f'<div class="value" style="color:#f87171;">{result["baseline_prob_fake"]*100:.1f}%</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            robust = "Robust" if not result["flipped"] else "Fragile"
            robust_color = "#34d399" if not result["flipped"] else "#f59e0b"
            st.markdown(
                f'<div class="metric-card"><h3>Robustness</h3>'
                f'<div class="value" style="color:{robust_color};">{robust}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Word-by-word contribution")
        render_word_chips(result["impacts"])

        st.plotly_chart(render_gauge(result["baseline_prob_real"]), use_container_width=True)

        left, right = st.columns([1, 1])
        with left:
            st.markdown("#### Top contributing words")
            render_top_words(result["top_words"])

        with right:
            st.markdown("#### Flip analysis")
            if result["flipped"]:
                st.warning(
                    f"Removing just **{result['removed_words']}** flips the prediction "
                    f"(new FAKE probability: {result['new_fake_prob']*100:.1f}%). "
                    "The label is fragile for this input."
                )
            else:
                st.success(
                    f"Could not flip the prediction by word removal alone "
                    f"(tried removing: {result['removed_words']}). "
                    "The label is fairly robust for this input."
                )

    st.markdown("---")
    st.caption(
        "Built on a model trained on combined Kaggle EMSCAD + Hugging Face fake-job-posting data. "
        "Explanations use a leave-one-word-out method — model-agnostic, no external XAI library required."
    )


if __name__ == "__main__":
    main()
