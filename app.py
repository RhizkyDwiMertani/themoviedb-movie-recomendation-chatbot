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
st.caption("⚡ Optimized Hybrid TF-IDF + BERT | Fast & Bilingual")

# =====================================================
# SIDEBAR HELP
# =====================================================
with st.sidebar:
    st.markdown("## 🤖 Cara Bertanya")
    st.markdown("""
**Mood**
- saya sedang sedih
- aku lagi senang

**Genre**
- rekomendasi horror
- film action

**Populer / Random**
- film terbaik
- random comedy

**Deskripsi**
- film tentang monster besar
- movie about giant robots
""")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("tmdb_movies.csv")
    df = df.drop_duplicates(subset="title")

    for col in ["title", "overview", "genres"]:
        df[col] = df[col].fillna("")

    for col in ["vote_count", "vote_average"]:
        df[col] = df[col].fillna(0)

    df["search_text"] = (
        df["title"].str.lower() + " " +
        df["overview"].str.lower().str[:300]
    )

    return df.reset_index(drop=True)

df = load_data()

# =====================================================
# LANGUAGE & TRANSLATION
# =====================================================
INDO_WORDS = ["film","tentang","saya","ingin","sedih","senang","rekomendasi"]

def is_indonesian(text):
    return any(w in text for w in INDO_WORDS)

@st.cache_data(show_spinner=False)
def translate_cached(text):
    try:
        return GoogleTranslator(source="id", target="en").translate(text)
    except:
        return text

# =====================================================
# TF-IDF
# =====================================================
@st.cache_resource
def build_tfidf(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
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
# BERT (FAST LOAD)
# =====================================================
@st.cache_resource
def load_bert():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_bert_embeddings():
    return np.load("bert_embeddings.npy")

bert_model = load_bert()
bert_embeddings = load_bert_embeddings()

def bert_search(query, n=5):
    vec = bert_model.encode([query], show_progress_bar=False)
    scores = cosine_similarity(vec, bert_embeddings)[0]
    idx = np.argsort(scores)[::-1][:n]
    return df.iloc[idx]

# =====================================================
# WEIGHTED RATING
# =====================================================
C = df["vote_average"].mean()
MIN_VOTES = df["vote_count"].quantile(0.70)

def weighted_rating(row):
    v, R = row["vote_count"], row["vote_average"]
    return (v/(v+MIN_VOTES))*R + (MIN_VOTES/(v+MIN_VOTES))*C

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
    "happy": ["comedy","family"],
    "senang": ["comedy","family"],
    "sad": ["drama","romance"],
    "sedih": ["drama","romance"],
    "seram": ["horror"],
    "scary": ["horror"]
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

def detect_intent(text):
    if any(k in text for k in ["random","acak"]):
        return "random"
    if any(k in text for k in ["rekomendasi","terbaik","popular"]):
        return "popular"
    return "normal"

# =====================================================
# UI CARD
# =====================================================
def movie_card(row):
    return f"""
<div style="padding:14px;border-radius:12px;background:#1c1f26;margin-bottom:12px">
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

user_input = st.chat_input("Ask movie recommendations...")

# =====================================================
# CHAT LOGIC
# =====================================================
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append({"role":"user","content":user_input})

    text = user_input.lower()
    genre = extract_genre(text)
    mood = extract_mood(text)
    intent = detect_intent(text)

    # 1. Mood + Genre
    if mood and genre:
        mask = df["genres"].str.lower().str.contains(genre, na=False)
        mood_mask = False
        for g in MOODS[mood]:
            mood_mask |= df["genres"].str.lower().str.contains(g, na=False)
        movies = top_movies(mask & mood_mask)
        response = f"🎭 **{genre.title()} movies for '{mood}' mood**\n\n"

    # 2. Mood
    elif mood:
        mood_mask = False
        for g in MOODS[mood]:
            mood_mask |= df["genres"].str.lower().str.contains(g, na=False)
        movies = top_movies(mood_mask)
        response = f"🎭 **Movies for '{mood}' mood**\n\n"

    # 3. Popular
    elif intent == "popular":
        movies = top_movies()
        response = "⭐ **Top Recommended Movies**\n\n"

    # 4. Random
    elif intent == "random":
        data = df
        if genre:
            data = data[data["genres"].str.lower().str.contains(genre, na=False)]
        movies = data.sample(min(5, len(data)))
        response = "🎲 **Random Movies**\n\n"

    # 5. TF-IDF → BERT fallback
    else:
        query = translate_cached(text) if is_indonesian(text) else text
        movies = tfidf_search(query)

        if movies["vote_count"].sum() < 50:
            movies = bert_search(query)

        response = "🔍 **Movies matching your request**\n\n"

    for _, r in movies.iterrows():
        response += movie_card(r)

    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    st.session_state.messages.append({"role":"assistant","content":response})
