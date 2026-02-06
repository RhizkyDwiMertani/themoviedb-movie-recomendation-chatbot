import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🎬 Movie Recommendation Chatbot",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation Chatbot")
st.caption("Hybrid TF-IDF + BERT | Mood, Genre, Popular, Random | Bilingual")

# =====================================================
# SIDEBAR HELP
# =====================================================
with st.sidebar:
    st.markdown("## 🤖 Cara Bertanya")

    st.markdown("### 🎭 Mood")
    st.markdown("""
- `saya sedang sedih`
- `aku lagi senang`
""")

    st.markdown("### 🎬 Genre")
    st.markdown("""
- `rekomendasi horror`
- `film action`
""")

    st.markdown("### 🔀 Mood + Genre")
    st.markdown("""
- `saya sedang sedih ingin horror`
- `lagi senang tapi mau nonton comedy`
""")

    st.markdown("### ⭐ Populer / 🎲 Random")
    st.markdown("""
- `rekomendasi film terbaik`
- `random horror`
""")

    st.markdown("### 🔍 Deskripsi Bebas")
    st.markdown("""
- `film tentang titan`
- `movie about giant monsters`
""")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_movies.csv")
    df = df.drop_duplicates(subset="title")

    for col in ["title", "overview", "genres"]:
        df[col] = df[col].fillna("")

    for col in ["vote_count", "vote_average"]:
        df[col] = df[col].fillna(0)

    df["search_text"] = (
        df["title"].str.lower() + " " + df["overview"].str.lower()
    )

    return df.reset_index(drop=True)

df = load_data()

# =====================================================
# LANGUAGE & TRANSLATION
# =====================================================
INDO_WORDS = ["film", "tentang", "saya", "ingin", "sedih", "senang", "rekomendasi"]

def is_indonesian(text):
    return any(w in text for w in INDO_WORDS)

def translate_to_english(text):
    try:
        return GoogleTranslator(source="id", target="en").translate(text)
    except:
        return text

# =====================================================
# TF-IDF (TITLE + OVERVIEW)
# =====================================================
@st.cache_resource
def build_tfidf(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        stop_words="english"
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

tfidf_model, tfidf_matrix = build_tfidf(df["search_text"].tolist())

def tfidf_search(query, n=5):
    vec = tfidf_model.transform([query])
    scores = cosine_similarity(vec, tfidf_matrix)[0]
    idx = np.argsort(scores)[::-1][:n]
    return df.iloc[idx]

# =====================================================
# BERT (FALLBACK)
# =====================================================
@st.cache_resource
def load_bert():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

bert_model = load_bert()
bert_embeddings = bert_model.encode(df["search_text"].tolist(), show_progress_bar=False)

def bert_search(query, n=5):
    vec = bert_model.encode([query])
    scores = cosine_similarity(vec, bert_embeddings)[0]
    idx = np.argsort(scores)[::-1][:n]
    return df.iloc[idx]

# =====================================================
# WEIGHTED RATING
# =====================================================
C = df["vote_average"].mean()
MIN_VOTES = df["vote_count"].quantile(0.70)

def weighted_rating(row):
    v = row["vote_count"]
    R = row["vote_average"]
    return (v / (v + MIN_VOTES)) * R + (MIN_VOTES / (v + MIN_VOTES)) * C

def top_movies(mask=None, n=5):
    data = df.copy()
    if mask is not None:
        data = data[mask]

    data = data[data["vote_count"] >= 30]
    if data.empty:
        return None

    data["score"] = data.apply(weighted_rating, axis=1)
    return data.sort_values("score", ascending=False).head(n)

# =====================================================
# GENRE & MOOD
# =====================================================
GENRES = [
    "action","adventure","animation","comedy","crime","drama",
    "family","fantasy","horror","romance","science fiction","thriller"
]

MOODS = {
    "happy": ["comedy", "family"],
    "senang": ["comedy", "family"],
    "sad": ["drama", "romance"],
    "sedih": ["drama", "romance"],
    "scary": ["horror"],
    "seram": ["horror"]
}

def extract_genre(text):
    for g in GENRES:
        if g in text:
            return g
    return None

def extract_mood(text):
    for m in MOODS:
        if m in text:
            return m
    return None

# =====================================================
# INTENT DETECTION
# =====================================================
def detect_intent(text):
    if any(k in text for k in ["random", "acak"]):
        return "random"
    if any(k in text for k in ["rekomendasi", "recommended", "terbaik", "popular"]):
        return "popular"
    return "normal"

# =====================================================
# UI CARD
# =====================================================
def movie_card(row):
    return f"""
<div style="padding:15px;border-radius:12px;background:#1c1f26;margin-bottom:12px;">
<h4>🎬 {row['title']}</h4>
<p>⭐ {row['vote_average']:.1f} | 👥 {int(row['vote_count'])}</p>
<p>🎭 {row['genres']}</p>
</div>
"""

# =====================================================
# CHAT STATE
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

user_input = st.chat_input("Ask in Indonesian or English...")

# =====================================================
# CHAT LOGIC (INTENT ROUTER)
# =====================================================
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    text = user_input.lower()
    genre = extract_genre(text)
    mood = extract_mood(text)
    intent = detect_intent(text)

    # 1️⃣ MOOD + GENRE
    if mood and genre:
        mask = (
            df["genres"].str.lower().str.contains(genre, na=False)
        )
        mood_mask = False
        for g in MOODS[mood]:
            mood_mask |= df["genres"].str.lower().str.contains(g, na=False)
        movies = top_movies(mask & mood_mask)
        response = f"🎬 **{genre.title()} movies for '{mood}' mood**\n\n"

    # 2️⃣ MOOD ONLY
    elif mood:
        mood_mask = False
        for g in MOODS[mood]:
            mood_mask |= df["genres"].str.lower().str.contains(g, na=False)
        movies = top_movies(mood_mask)
        response = f"🎭 **Movies for '{mood}' mood**\n\n"

    # 3️⃣ POPULAR
    elif intent == "popular":
        mask = df["genres"].str.lower().str.contains(genre, na=False) if genre else None
        movies = top_movies(mask)
        response = "⭐ **Top Recommended Movies**\n\n"

    # 4️⃣ RANDOM
    elif intent == "random":
        data = df
        if genre:
            data = df[df["genres"].str.lower().str.contains(genre, na=False)]
        movies = data.sample(min(5, len(data)))
        response = "🎲 **Random Movies**\n\n"

    # 5️⃣ DESCRIPTIVE (TF-IDF)
    elif len(text.split()) >= 5:
        query = translate_to_english(text) if is_indonesian(text) else text
        movies = tfidf_search(query)
        response = "🔍 **Movies matching your description**\n\n"

    # 6️⃣ FALLBACK (BERT)
    else:
        query = translate_to_english(text) if is_indonesian(text) else text
        movies = bert_search(query)
        response = "🧠 **Movies matching your request**\n\n"

    for _, r in movies.iterrows():
        response += movie_card(r)

    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
