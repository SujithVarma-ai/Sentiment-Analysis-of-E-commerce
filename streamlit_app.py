import streamlit as st
import io
import os
import re
import pickle

@st.cache_resource
def load_model():
    model = pickle.load(open("logistic_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
APP_TITLE = "🛒 E-Commerce Sentiment Analyzer"
DATASET_PATH = "data/amazon_reviews.csv"

SENTIMENT_COLORS = {
    "Positive": "#00C851",
    "Neutral":  "#ffbb33",
    "Negative": "#ff4444",
}

SENTIMENT_EMOJIS = {
    "Positive": "😊",
    "Neutral":  "😐",
    "Negative": "😠",
}

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(ellipse at center, rgba(100, 100, 255, 0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.05); opacity: 1; }
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 0.5rem 0;
        position: relative;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: rgba(255,255,255,0.7);
        margin: 0;
        position: relative;
        font-weight: 300;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255,255,255,0.5);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* Sentiment result cards */
    .sentiment-result {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.25);
        position: relative;
        overflow: hidden;
    }
    .sentiment-result::after {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
    }
    .sentiment-positive {
        background: linear-gradient(145deg, #0a2e1a, #0d3d22);
        border: 1px solid rgba(0, 200, 81, 0.3);
    }
    .sentiment-positive::after { background: linear-gradient(90deg, #00C851, #00e676); }
    .sentiment-neutral {
        background: linear-gradient(145deg, #2e2a0a, #3d3a0d);
        border: 1px solid rgba(255, 187, 51, 0.3);
    }
    .sentiment-neutral::after { background: linear-gradient(90deg, #ffbb33, #ffca28); }
    .sentiment-negative {
        background: linear-gradient(145deg, #2e0a0a, #3d0d0d);
        border: 1px solid rgba(255, 68, 68, 0.3);
    }
    .sentiment-negative::after { background: linear-gradient(90deg, #ff4444, #ff5252); }

    .sentiment-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    .sentiment-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    .sentiment-confidence {
        font-size: 1rem;
        opacity: 0.7;
        font-weight: 300;
    }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }

    /* Info box */
    .info-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-left: 4px solid #7c3aed;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1rem 0;
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }

    /* Probability bars */
    .prob-bar-container {
        margin: 0.6rem 0;
    }
    .prob-bar-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 4px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .prob-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
    }
</style>
"""


# ─── NLTK SETUP ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_nltk_stopwords() -> set[str]:
    import nltk
    from nltk.corpus import stopwords
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    
    sw = set(stopwords.words("english"))
    
    # Remove crucial negations so sentiment meaning isn't lost
    negations = {"not", "no", "nor", "none", "don", "don't", "aren", "aren't", "isn", "isn't", 
                 "wasn", "wasn't", "weren", "weren't", "hasn", "hasn't", "haven", "haven't", "hadn", "hadn't",
                 "doesn", "doesn't", "didn", "didn't", "shouldn", "shouldn't", "wouldn", "wouldn't", "couldn", "couldn't",
                 "won", "won't", "can", "can't", "cannot", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
                 "against"}
                 
    return sw - negations


STOPWORDS = _load_nltk_stopwords()


# ─── TEXT CLEANING ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = [w for w in text.split() if w and (w not in STOPWORDS)]
    return " ".join(words)


# ─── SENTIMENT MAPPING ───────────────────────────────────────────────────────
def map_score_to_sentiment(score):
    """Map numeric score to sentiment label."""
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_dataset(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv("data/amazon_reviews.csv")

    df = df.dropna(subset=["content", "score"])
    df["content"] = df["content"].astype(str)
    df["sentiment"] = df["score"].apply(map_score_to_sentiment)
    df["review_length"] = df["content"].apply(len)

    return df


# ─── MODEL TRAINING ──────────────────────────────────────────────────────────
@dataclass
class TrainResult:
    pipeline: Pipeline
    test_accuracy: float
    cm: np.ndarray
    labels: list
    report: str

#Logistic Regression

def build_pipeline(max_features: int, ngram_max: int, c: float) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, ngram_max),
                    min_df=2,
                    max_df=0.95,
                    preprocessor=clean_text,
                ),
            ),
            ("clf", LogisticRegression(max_iter=4000, n_jobs=-1, C=c)),
        ]
    )


def train_model(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float,
    random_state: int,
    max_features: int,
    ngram_max: int,
    c: float,
) -> TrainResult:
    data = df[[text_col, label_col]].dropna()
    data[text_col] = data[text_col].astype(str)

    X = data[text_col]
    y = data[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    pipe = build_pipeline(max_features=max_features, ngram_max=ngram_max, c=c)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    labels = sorted(pd.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels)

    return TrainResult(pipeline=pipe, test_accuracy=acc, cm=cm, labels=labels, report=report)


# ─── CHART HELPERS ────────────────────────────────────────────────────────────
def create_donut_chart(probabilities: dict):
    """Create beautiful donut chart for sentiment probabilities."""
    labels = list(probabilities.keys())
    values = [v * 100 for v in probabilities.values()]
    colors = [SENTIMENT_COLORS.get(l, "#888") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0.3)', width=2)),
        textinfo='label+percent',
        textfont=dict(size=14, family="Inter", color="white"),
        hovertemplate="<b>%{label}</b><br>Confidence: %{value:.1f}%<extra></extra>",
        pull=[0.03] * len(labels),
    )])

    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20),
        height=320,
        annotations=[dict(
            text="<b>Confidence</b>",
            x=0.5, y=0.5,
            font=dict(size=16, color="white", family="Inter"),
            showarrow=False,
        )],
    )
    return fig


def create_sentiment_distribution_chart(df: pd.DataFrame):
    """Create bar chart for sentiment distribution."""
    counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    colors = [SENTIMENT_COLORS[s] for s in counts.index]

    fig = go.Figure(data=[go.Bar(
        x=counts.index,
        y=counts.values,
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=8,
        ),
        text=[f"{v:,}" for v in counts.values],
        textposition='outside',
        textfont=dict(size=14, color="white", family="Inter"),
        hovertemplate="<b>%{x}</b><br>Count: %{y:,}<extra></extra>",
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            color='rgba(255,255,255,0.7)',
            tickfont=dict(size=13, family="Inter"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            color='rgba(255,255,255,0.7)',
            tickfont=dict(size=12, family="Inter"),
        ),
        margin=dict(t=30, b=40, l=50, r=20),
        height=380,
    )
    return fig


def create_sentiment_pie(df: pd.DataFrame):
    """Create pie chart for sentiment percentage split."""
    counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"]).fillna(0)
    colors = [SENTIMENT_COLORS[s] for s in counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0.3)', width=2)),
        textinfo='label+percent',
        textfont=dict(size=13, family="Inter", color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    )])

    fig.update_layout(
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20),
        height=350,
    )
    return fig


def create_review_length_chart(df: pd.DataFrame):
    """Create histogram for review length distribution."""
    fig = go.Figure()

    for sentiment in ["Positive", "Neutral", "Negative"]:
        subset = df[df["sentiment"] == sentiment]
        fig.add_trace(go.Histogram(
            x=subset["review_length"],
            name=sentiment,
            marker_color=SENTIMENT_COLORS[sentiment],
            opacity=0.7,
            nbinsx=50,
        ))

    fig.update_layout(
        barmode='overlay',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=dict(text="Review Length (characters)", font=dict(size=12, family="Inter")),
            showgrid=False,
            color='rgba(255,255,255,0.7)',
            tickfont=dict(size=11, family="Inter"),
        ),
        yaxis=dict(
            title=dict(text="Frequency", font=dict(size=12, family="Inter")),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            color='rgba(255,255,255,0.7)',
            tickfont=dict(size=11, family="Inter"),
        ),
        legend=dict(
            font=dict(color="white", family="Inter"),
            bgcolor='rgba(0,0,0,0)',
        ),
        margin=dict(t=20, b=50, l=60, r=20),
        height=380,
    )
    return fig


def create_confusion_matrix_chart(cm, labels):
    """Create heatmap for confusion matrix."""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[
            [0, '#1a1a2e'],
            [0.25, '#16213e'],
            [0.5, '#302b63'],
            [0.75, '#7c3aed'],
            [1, '#a78bfa'],
        ],
        text=cm,
        texttemplate="%{text}",
        textfont=dict(size=16, color="white", family="Inter"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False,
    ))

    fig.update_layout(
        xaxis=dict(
            title=dict(text="Predicted", font=dict(size=13, family="Inter")),
            color='rgba(255,255,255,0.7)',
            tickfont=dict(size=12, family="Inter"),
        ),
        yaxis=dict(
            title=dict(text="Actual", font=dict(size=13, family="Inter")),
            color='rgba(255,255,255,0.7)',
            tickfont=dict(size=12, family="Inter"),
            autorange='reversed',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=60, l=60, r=20),
        height=380,
    )
    return fig


# ─── MAIN APP ─────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="E-Commerce Sentiment Analyzer",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Hero Section ──
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">🛒 E-Commerce Sentiment Analyzer</div>
            <div class="hero-subtitle">
                NLP-Powered Sentiment Analysis of Amazon Shopping Reviews &nbsp;•&nbsp;
                Powered by TF-IDF + Logistic Regression
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Load Data ──
    with st.spinner("Loading dataset..."):
        df = load_dataset(DATASET_PATH)

    # ── Top Metrics ──
    total = len(df)
    pos_count = len(df[df["sentiment"] == "Positive"])
    neu_count = len(df[df["sentiment"] == "Neutral"])
    neg_count = len(df[df["sentiment"] == "Negative"])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total:,}</div>
                <div class="metric-label">Total Reviews</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #00C851">{pos_count:,}</div>
                <div class="metric-label">Positive</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ffbb33">{neu_count:,}</div>
                <div class="metric-label">Neutral</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: #ff4444">{neg_count:,}</div>
                <div class="metric-label">Negative</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    tabs = st.tabs([
        "📊  Dashboard",
        "⚙️  Train Model",
        "🔮  Predict Sentiment",
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — DASHBOARD / EDA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[0]:
        st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

        eda1, eda2 = st.columns(2)
        with eda1:
            st.markdown("##### Sentiment Distribution")
            st.plotly_chart(create_sentiment_distribution_chart(df), use_container_width=True)
        with eda2:
            st.markdown("##### Sentiment Percentage Split")
            st.plotly_chart(create_sentiment_pie(df), use_container_width=True)

        st.markdown("##### Review Length by Sentiment")
        st.plotly_chart(create_review_length_chart(df), use_container_width=True)

        # Word cloud
        with st.expander("☁️ Show Word Cloud (may take a moment)"):
            wc_col = st.selectbox("Generate word cloud for:", ["All", "Positive", "Neutral", "Negative"], key="wc_filter")
            if st.button("Generate Word Cloud", key="btn_wc"):
                try:
                    from wordcloud import WordCloud
                    import matplotlib.pyplot as plt

                    subset = df if wc_col == "All" else df[df["sentiment"] == wc_col]
                    text = " ".join(subset["content"].head(5000).tolist())
                    wc = WordCloud(
                        background_color="#0e1117",
                        colormap="Set2",
                        max_words=100,
                        width=900,
                        height=400,
                    ).generate(text)
                    fig, ax = plt.subplots(figsize=(12, 5))
                    fig.patch.set_facecolor('#0e1117')
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Word cloud failed: {e}")

        # Data preview
        with st.expander("🗂️ Preview Raw Data"):
            st.dataframe(
                df[["userName", "content", "score", "sentiment", "review_length"]].head(50),
                use_container_width=True,
                height=400,
            )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — TRAIN MODEL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[1]:
        st.markdown('<div class="section-header">⚙️ Train Sentiment Model</div>', unsafe_allow_html=True)

        st.markdown("""
            <div class="info-box">
                <strong>How it works:</strong> The model uses TF-IDF vectorization to convert review text into 
                numerical features, then trains a Logistic Regression classifier to predict sentiment 
                (Positive / Neutral / Negative). Adjust the hyperparameters below and hit <b>Train</b>.
            </div>
        """, unsafe_allow_html=True)

        # Hyperparameters
        h1, h2, h3 = st.columns(3)
        with h1:
            max_features = st.slider("Max TF-IDF Features", 2000, 30000, 7000, step=1000, key="hp_maxf")
        with h2:
            ngram_max = st.selectbox("Max n-gram", [1, 2, 3], index=1, key="hp_ngram")
        with h3:
            c = st.slider("Regularization (C)", 0.1, 10.0, 1.0, step=0.1, key="hp_c")

        s1, s2 = st.columns(2)
        with s1:
            test_size = st.slider("Test split ratio", 0.1, 0.5, 0.3, step=0.05, key="hp_test")
        with s2:
            random_state = st.number_input("Random seed", value=42, step=1, key="hp_seed")

        if st.button("🚀  Train Model", type="primary", use_container_width=True):
            with st.spinner("Training model... this may take a minute for large datasets"):
                result = train_model(
                    df=df,
                    text_col="content",
                    label_col="sentiment",
                    test_size=float(test_size),
                    random_state=int(random_state),
                    max_features=int(max_features),
                    ngram_max=int(ngram_max),
                    c=float(c),
                )

            st.session_state["model"] = result.pipeline
            st.session_state["model_labels"] = result.labels

            # Success metrics
            st.success(f"✅ Model trained successfully!")

            acc_col, cm_col = st.columns([1, 2])
            with acc_col:
                st.markdown(f"""
                    <div class="metric-card" style="margin-top:1rem">
                        <div class="metric-value" style="color: #7c3aed">{result.test_accuracy:.1%}</div>
                        <div class="metric-label">Test Accuracy</div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Classification Report:**")
                st.code(result.report, language="text")

            with cm_col:
                st.markdown("##### Confusion Matrix")
                st.plotly_chart(
                    create_confusion_matrix_chart(result.cm, result.labels),
                    use_container_width=True,
                )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3 — PREDICT SENTIMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tabs[2]:
        st.markdown('<div class="section-header">🔮 Predict Sentiment</div>', unsafe_allow_html=True)

        model = st.session_state.get("model")

        if model is None:
            st.warning("⚠️ No model loaded. Please train a model in the **⚙️ Train Model** tab first.")
        else:
            st.markdown("""
                <div class="info-box">
                    Enter a product review or any e-commerce feedback below and click 
                    <b>Analyze Sentiment</b> to predict whether it's Positive, Neutral, or Negative.
                </div>
            """, unsafe_allow_html=True)

            # Text input
            review_text = st.text_area(
                "✍️ Enter your review text:",
                height=160,
                placeholder="e.g. 'This product is amazing! Great quality and fast delivery. Highly recommend it to everyone.'",
                key="predict_input",
            )

            if st.button("🔍  Analyze Sentiment", type="primary", use_container_width=True):
                if not review_text.strip():
                    st.error("Please enter some text to analyze.")
                else:
                    # Predict
                    prediction = model.predict([review_text])[0]
                    probabilities_raw = model.predict_proba([review_text])[0]
                    class_labels = model.classes_

                    prob_dict = {label: float(prob) for label, prob in zip(class_labels, probabilities_raw)}

                    # Ensure all sentiments present
                    for s in ["Positive", "Neutral", "Negative"]:
                        if s not in prob_dict:
                            prob_dict[s] = 0.0

                    confidence = prob_dict[prediction] * 100
                    emoji = SENTIMENT_EMOJIS.get(prediction, "🤔")
                    css_class = f"sentiment-{prediction.lower()}"

                    # Result card
                    st.markdown(f"""
                        <div class="sentiment-result {css_class}">
                            <div class="sentiment-emoji">{emoji}</div>
                            <div class="sentiment-label" style="color: {SENTIMENT_COLORS[prediction]}">{prediction}</div>
                            <div class="sentiment-confidence">Confidence: {confidence:.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Probability breakdown
                    st.markdown("<br>", unsafe_allow_html=True)

                    chart_col, bars_col = st.columns([1, 1])

                    with chart_col:
                        st.markdown("##### Confidence Distribution")
                        st.plotly_chart(create_donut_chart(prob_dict), use_container_width=True)

                    with bars_col:
                        st.markdown("##### Probability Breakdown")
                        st.markdown("<br>", unsafe_allow_html=True)

                        for label in ["Positive", "Neutral", "Negative"]:
                            pct = prob_dict.get(label, 0) * 100
                            color = SENTIMENT_COLORS[label]
                            st.markdown(f"""
                                <div class="prob-bar-container">
                                    <div class="prob-bar-label">
                                        <span>{SENTIMENT_EMOJIS[label]} {label}</span>
                                        <span style="color: {color}; font-weight: 700;">{pct:.1f}%</span>
                                    </div>
                                    <div class="prob-bar-bg">
                                        <div class="prob-bar-fill" style="width: {pct}%; background: {color};"></div>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                        st.markdown("<br>", unsafe_allow_html=True)


    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:rgba(255,255,255,0.3);font-size:0.8rem;padding:1rem 0;'>"
        "SPL Project — Sentiment Analysis of E-Commerce using NLP &nbsp;•&nbsp; Built with Streamlit & Scikit-learn"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()