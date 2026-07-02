"""
Eco-Verify — Streamlit Frontend
================================
Interactive web dashboard for the Comparative Gap Analysis Tool for
Corporate and Consumer Social Responsibility using ML-based Topic
Modeling and NLP.

Run with:
    streamlit run app.py
"""

import re
import random
import io

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Eco-Verify", page_icon="🌿", layout="wide")


@st.cache_resource
def ensure_nltk():
    for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)


ensure_nltk()

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
CUSTOM_STOPWORDS = {"brand", "product", "company", "would", "really", "get", "im",
                     "need", "willing", "find", "buy", "read", "making", "us",
                     "instead", "much", "think", "switched", "glad", "loved",
                     "behind", "real", "come", "every", "wish", "still", "even",
                     "always", "clear", "nice", "like", "look", "looks"}
STOP_WORDS = STOP_WORDS.union(CUSTOM_STOPWORDS)

# ---------------------------------------------------------------------------
# SYNTHETIC DATA GENERATION (used when no file is uploaded)
# ---------------------------------------------------------------------------
PRODUCTS = ["sneakers", "t-shirt", "shampoo bottle", "laptop", "coffee brand",
            "smartphone", "jacket", "backpack", "skincare set", "grocery brand",
            "furniture set", "energy drink", "detergent", "handbag", "headphones"]

CONSUMER_TEMPLATES = {
    "packaging_waste": [
        "The {p} came wrapped in so much plastic, it felt so wasteful and unnecessary.",
        "I love that this {p} uses recyclable cardboard instead of plastic packaging.",
        "Way too much single-use plastic packaging for a small {p}, really disappointed.",
        "The {p} box was biodegradable and compostable, which I really appreciated.",
        "Why does a {p} need three layers of plastic wrap? Terrible for the environment.",
    ],
    "labor_practices": [
        "I read that workers making this {p} are underpaid, I won't buy it again.",
        "Glad to see this {p} brand is fair-trade certified and pays workers fairly.",
        "There are rumors of poor factory conditions behind this {p}, very concerning.",
        "The company publishes factory audit reports for their {p}, that builds trust.",
        "No transparency about who makes this {p} or under what working conditions.",
    ],
    "greenwashing_skepticism": [
        "This {p} brand claims to be eco-friendly but I think it's just greenwashing.",
        "Not convinced their 'sustainable' {p} claims are backed by any real proof.",
        "They slap a green leaf logo on the {p} but nothing about it is actually sustainable.",
        "I appreciate that the {p} brand backs its eco claims with third-party certification.",
        "Marketing calls this {p} 'green' but the ingredients list says otherwise.",
    ],
    "carbon_footprint": [
        "The {p} ships from overseas which must have a huge carbon footprint.",
        "Loved that this {p} brand offsets carbon emissions from shipping.",
        "No mention anywhere of the carbon footprint of producing this {p}.",
        "This {p} company publishes its carbon emissions every year, very transparent.",
        "Shipping this {p} internationally seems terrible for emissions and climate.",
    ],
    "ethical_sourcing": [
        "Happy to see the {p} uses ethically sourced, cruelty-free materials.",
        "Not sure where the raw materials for this {p} even come from, no info given.",
        "This {p} brand sources materials responsibly and shows proof on their site.",
        "The {p} ingredients might come from unethical suppliers, hard to verify.",
        "I wish there was clearer sourcing information printed on the {p} label.",
    ],
    "recycling_biodegradable": [
        "This {p} is fully recyclable and the company even offers a take-back program.",
        "None of the {p} materials seem recyclable, it will just end up in a landfill.",
        "The {p} packaging is biodegradable which is a nice environmentally friendly touch.",
        "I tried to recycle the {p} container but my local center won't accept it.",
        "Great that this {p} brand uses recycled materials in their products.",
    ],
    "price_of_ethical": [
        "Sustainable options for this {p} are always so much more expensive than regular ones.",
        "Willing to pay a bit more for the {p} because it's ethically made.",
        "Ethical {p} brands need to be more affordable for everyday consumers like me.",
        "The eco-friendly version of this {p} costs almost double, not everyone can afford that.",
        "Price shouldn't be a barrier to buying a socially responsible {p}.",
    ],
    "corporate_transparency": [
        "This {p} company publishes a detailed sustainability report every year, very transparent.",
        "I can't find any information about this {p} company's social responsibility efforts.",
        "They openly share supply chain data for the {p}, which builds a lot of trust.",
        "This {p} brand hides behind vague statements instead of real transparency.",
        "Would love more transparency from this {p} brand about their environmental impact.",
    ],
    "animal_testing": [
        "Glad this {p} is certified cruelty-free and never tested on animals.",
        "Not clear if this {p} brand tests on animals, wish it was labeled clearly.",
        "Switched to this {p} because it's vegan and cruelty-free certified.",
        "Disappointed to learn this {p} brand still tests on animals in some markets.",
    ],
}

COMPANIES = ["GreenLeaf Corp", "Nimbus Retail", "BrightPath Foods", "Solara Apparel",
             "UrbanCraft Goods", "Cobalt Electronics", "PureWell Cosmetics", "Trailhead Outdoors"]

CORPORATE_TEMPLATES = {
    "carbon_neutrality_pledge": [
        "{c} is committed to achieving net-zero carbon emissions by 2040 across our global operations.",
        "{c} has pledged to offset 100% of its shipping emissions through verified carbon credit programs.",
        "Our roadmap at {c} outlines a science-based target to cut greenhouse gas emissions by 50% this decade.",
    ],
    "philanthropy": [
        "{c} donated over one million dollars to community environmental education programs this year.",
        "Through the {c} Foundation, we support local charities focused on clean water access.",
        "{c} employees volunteered over ten thousand hours to local sustainability initiatives in 2025.",
    ],
    "diversity_inclusion": [
        "{c} is proud of its diverse workforce and ongoing commitment to equitable hiring practices.",
        "Our leadership team at {c} reflects a broad range of backgrounds and lived experiences.",
        "{c} launched new mentorship programs to support underrepresented employees company-wide.",
    ],
    "sustainability_pledge_vague": [
        "{c} believes in building a better, greener future for generations to come.",
        "Sustainability is at the heart of everything {c} does, from sourcing to shipping.",
        "{c} strives to minimize its environmental footprint through continuous innovation.",
    ],
    "certifications": [
        "{c} products are now certified under leading global sustainability and fair-trade standards.",
        "{c} has earned recognition from independent auditors for responsible sourcing practices.",
        "This year {c} achieved B-Corp style certification for its social and environmental performance.",
    ],
    "supply_chain": [
        "{c} works closely with suppliers to ensure fair wages and safe working conditions.",
        "Our supply chain partners at {c} undergo regular audits to verify ethical labor standards.",
        "{c} is expanding supplier transparency initiatives to trace materials back to their origin.",
    ],
}


def generate_consumer_reviews(seed=42):
    random.seed(seed)
    rows, review_id = [], 1
    for theme, temps in CONSUMER_TEMPLATES.items():
        for temp in temps:
            for _ in range(8):
                p = random.choice(PRODUCTS)
                rows.append({"review_id": review_id, "product": p, "theme_true": theme,
                             "review_text": temp.format(p=p)})
                review_id += 1
    random.shuffle(rows)
    return pd.DataFrame(rows)


def generate_corporate_statements(seed=42):
    random.seed(seed)
    rows, stmt_id = [], 1
    for theme, temps in CORPORATE_TEMPLATES.items():
        for temp in temps:
            for c in COMPANIES:
                rows.append({"statement_id": stmt_id, "company": c, "theme_true": theme,
                             "statement_text": temp.format(c=c)})
                stmt_id += 1
    random.shuffle(rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NLP PIPELINE
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str) -> str:
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def build_tfidf(corpus, max_features=500):
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                                  min_df=2, max_df=0.9)
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def run_topic_model(tfidf_matrix, vectorizer, n_topics, method="lda"):
    if method == "lda":
        model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    else:
        model = NMF(n_components=n_topics, random_state=42, max_iter=500)
    doc_topic = model.fit_transform(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for idx, comp in enumerate(model.components_):
        top_indices = comp.argsort()[-10:][::-1]
        topics[f"Topic {idx+1}"] = [feature_names[i] for i in top_indices]
    return model, doc_topic, topics


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 0


def make_wordcloud_fig(text_series, title):
    text = " ".join(text_series) if len(text_series) else "no data"
    wc = WordCloud(width=900, height=450, background_color="white",
                    colormap="viridis").generate(text)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=13)
    return fig


def make_topic_distribution_fig(df, title):
    counts = df["dominant_topic"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.values, color="#c0392b")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Documents")
    ax.set_title(title)
    return fig


def make_keywords_fig(topics_dict, title):
    fig, axes = plt.subplots(len(topics_dict), 1, figsize=(7, 2.0 * len(topics_dict)))
    if len(topics_dict) == 1:
        axes = [axes]
    for ax, (topic, words) in zip(axes, topics_dict.items()):
        ax.barh(words[::-1], range(1, len(words) + 1), color="#2980b9")
        ax.set_title(topic, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SIDEBAR — CONTROLS
# ---------------------------------------------------------------------------
st.sidebar.title("🌿 Eco-Verify")
st.sidebar.caption("Comparative Gap Analysis Tool for Corporate & Consumer Social Responsibility")

st.sidebar.header("1. Data Source")
data_mode = st.sidebar.radio(
    "Choose data source",
    ["Use built-in sample data", "Upload my own CSV files"],
    index=0,
)

consumer_file = corporate_file = None
if data_mode == "Upload my own CSV files":
    st.sidebar.markdown("**Consumer reviews CSV** needs a `review_text` column.")
    consumer_file = st.sidebar.file_uploader("Consumer reviews CSV", type="csv", key="cons")
    st.sidebar.markdown("**Corporate statements CSV** needs a `statement_text` column.")
    corporate_file = st.sidebar.file_uploader("Corporate statements CSV", type="csv", key="corp")

st.sidebar.header("2. Model Settings")
n_topics = st.sidebar.slider("Number of topics", min_value=3, max_value=10, value=6)
method = st.sidebar.selectbox("Topic modeling algorithm", ["LDA", "NMF"], index=0)
gap_threshold = st.sidebar.slider("Gap threshold (Jaccard overlap)", 0.0, 0.3, 0.08, 0.01)

run_button = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------------------------------
st.title("🌿 Eco-Verify")
st.subheader("A Comparative Gap Analysis Tool for Corporate and Consumer Social Responsibility")
st.caption("Using Machine Learning-Based Topic Modeling and NLP")

with st.expander("ℹ️ About this project", expanded=False):
    st.markdown("""
This tool analyzes **consumer reviews** and **corporate CSR statements** separately using NLP and
topic modeling (LDA/NMF), then performs a **gap analysis** to show which consumer concerns are
*not* being addressed in corporate messaging.

**Pipeline:** Data → Text Preprocessing (tokenize, stopword removal, lemmatize) → TF-IDF →
Topic Modeling → Visualization → Gap Analysis.
""")

if "ran" not in st.session_state:
    st.session_state.ran = False

if run_button:
    st.session_state.ran = True

if not st.session_state.ran:
    st.info("👈 Configure options in the sidebar and click **Run Analysis** to get started.")
    st.stop()

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
with st.spinner("Loading data..."):
    if data_mode == "Upload my own CSV files" and consumer_file and corporate_file:
        consumer_df = pd.read_csv(consumer_file)
        corporate_df = pd.read_csv(corporate_file)
        if "review_text" not in consumer_df.columns:
            st.error("Consumer CSV must have a 'review_text' column.")
            st.stop()
        if "statement_text" not in corporate_df.columns:
            st.error("Corporate CSV must have a 'statement_text' column.")
            st.stop()
    else:
        if data_mode == "Upload my own CSV files":
            st.warning("No files uploaded — falling back to built-in sample data.")
        consumer_df = generate_consumer_reviews()
        corporate_df = generate_corporate_statements()

col1, col2 = st.columns(2)
col1.metric("Consumer reviews loaded", len(consumer_df))
col2.metric("Corporate statements loaded", len(corporate_df))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data", "🧹 Preprocessing", "🔍 Consumer Topics", "🏢 Corporate Topics", "⚖️ Gap Analysis"]
)

with tab1:
    st.subheader("Consumer Reviews (sample)")
    st.dataframe(consumer_df.head(10), use_container_width=True)
    st.subheader("Corporate Statements (sample)")
    st.dataframe(corporate_df.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
with st.spinner("Preprocessing text (tokenizing, removing stopwords, lemmatizing)..."):
    consumer_df["clean_text"] = consumer_df["review_text"].apply(preprocess)
    corporate_df["clean_text"] = corporate_df["statement_text"].apply(preprocess)

with tab2:
    st.subheader("Before vs After Preprocessing")
    preview = consumer_df[["review_text", "clean_text"]].head(8)
    st.dataframe(preview, use_container_width=True)

# ---------------------------------------------------------------------------
# TF-IDF + TOPIC MODELING
# ---------------------------------------------------------------------------
with st.spinner("Running TF-IDF and topic modeling..."):
    consumer_vectorizer, consumer_tfidf = build_tfidf(consumer_df["clean_text"])
    corporate_vectorizer, corporate_tfidf = build_tfidf(corporate_df["clean_text"])

    method_key = method.lower()
    _, consumer_doc_topic, consumer_topics = run_topic_model(
        consumer_tfidf, consumer_vectorizer, n_topics, method=method_key)
    _, corporate_doc_topic, corporate_topics = run_topic_model(
        corporate_tfidf, corporate_vectorizer, n_topics, method=method_key)

    consumer_df["dominant_topic"] = consumer_doc_topic.argmax(axis=1) + 1
    corporate_df["dominant_topic"] = corporate_doc_topic.argmax(axis=1) + 1

with tab3:
    st.subheader("Consumer Topics")
    st.pyplot(make_wordcloud_fig(consumer_df["clean_text"], "Consumer Review Word Cloud"))
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(make_topic_distribution_fig(consumer_df, "Consumer Topic Distribution"))
    with c2:
        st.pyplot(make_keywords_fig(consumer_topics, "Top Keywords per Consumer Topic"))
    st.markdown("**Topic keywords:**")
    for t, words in consumer_topics.items():
        st.write(f"**{t}:** {', '.join(words)}")

with tab4:
    st.subheader("Corporate Topics")
    st.pyplot(make_wordcloud_fig(corporate_df["clean_text"], "Corporate CSR Statement Word Cloud"))
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(make_topic_distribution_fig(corporate_df, "Corporate Topic Distribution"))
    with c2:
        st.pyplot(make_keywords_fig(corporate_topics, "Top Keywords per Corporate Topic"))
    st.markdown("**Topic keywords:**")
    for t, words in corporate_topics.items():
        st.write(f"**{t}:** {', '.join(words)}")

# ---------------------------------------------------------------------------
# GAP ANALYSIS
# ---------------------------------------------------------------------------
gap_records = []
for c_topic, c_words in consumer_topics.items():
    best_match, best_score = None, 0
    for k_topic, k_words in corporate_topics.items():
        score = jaccard(c_words, k_words)
        if score > best_score:
            best_score, best_match = score, k_topic
    gap_records.append({
        "consumer_topic": c_topic,
        "consumer_keywords": ", ".join(c_words[:6]),
        "best_corporate_match": best_match,
        "overlap_score": round(best_score, 3),
        "gap_flag": "GAP" if best_score < gap_threshold else "ADDRESSED",
    })
gap_df = pd.DataFrame(gap_records).sort_values("overlap_score")

with tab5:
    st.subheader("Gap Analysis: Consumer Concerns vs Corporate CSR Messaging")
    n_gaps = (gap_df["gap_flag"] == "GAP").sum()
    st.metric("Unaddressed consumer concerns (GAPS)", f"{n_gaps} / {len(gap_df)}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#e74c3c" if g == "GAP" else "#27ae60" for g in gap_df["gap_flag"]]
    ax.barh(gap_df["consumer_topic"], gap_df["overlap_score"], color=colors)
    ax.axvline(gap_threshold, color="black", linestyle="--", linewidth=1, label="Gap threshold")
    ax.set_xlabel("Overlap Score with Closest Corporate Topic (Jaccard)")
    ax.set_title("Gap Analysis")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(gap_df, use_container_width=True)

    csv_buffer = io.StringIO()
    gap_df.to_csv(csv_buffer, index=False)
    st.download_button("⬇️ Download Gap Analysis CSV", csv_buffer.getvalue(),
                        file_name="gap_analysis.csv", mime="text/csv")

    st.markdown("""
    **How to read this:** each row is a topic consumers talk about. The bar shows how closely
    the *closest* corporate CSR topic matches it. Short red bars mean that consumer concern has
    **no real counterpart** in corporate messaging — a genuine communication gap.
    """)
