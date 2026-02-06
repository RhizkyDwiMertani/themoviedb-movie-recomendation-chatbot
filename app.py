import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rapidfuzz import process, fuzz
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
st.caption("Hybrid TF-IDF + BERT | Mood-aware | Anime-aware | Bilingual")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_movies.csv")
    df = df.drop_duplicates(subset="title")

    for col in ["genres", "overview", "original_language"]:
        df[col] = df[col].fillna("")
    for col in ["vote_count", "vote_average", "popularity"]:
        df[col] = df[col].fillna(0)

    # TF-IDF text (TITLE + OVERVIEW + GENRES)
    df["tfidf_text"] = df["title"] + " " + df["overview"] + " " + df["genres"]

    # BERT text
    df["bert_text"] = df["overview"] + " " + df["genres"]

    return df.reset_index(drop=True)

df = load_data()

# =====================================================
# TRANSLATION (SAFE)
# =====================================================
def auto_translate(text, src, dest):
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except:
        return text

def detect_language(text):
    indo_words = ["film", "tentang", "mirip", "seperti", "rekomendasi", "ingin", "sedih", "senang"]
    return "id" if any(w in text for w in indo_words) else "en"

# =====================================================
# TF-IDF MODEL (DESCRIPTIVE SEARCH)
# =====================================================
@st.cache_resource
def build_tfidf(texts):
    tfidf = TfidfVectorizer(
        stop_words=None,        # penting untuk Bahasa Indonesia
        ngram_range=(1, 2),
        max_features=15000
    )
    matrix = tfidf.fit_transform(texts)
    return tfidf, matrix

tfidf_model, tfidf_matrix = build_tfidf(df["tfidf_text"].tolist())

def tfidf_search(query, n=5):
    vec = tfidf_model.transform([query])
    scores = cosine_similarity(vec, tfidf_matrix)[0]
    idx = np.argsort(scores)[::-1][:n]
    return df.iloc[idx]

# =====================================================
# BERT MODEL (SEMANTIC FALLBACK)
# =====================================================
@st.cache_resource
def load_bert():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

bert_model = load_bert()

@st.cache_resource
def build_bert_embeddings(texts):
    return bert_model.encode(texts, show_progress_bar=False)

bert_embeddings = build_bert_embeddings(df["bert_text"].tolist())

def bert_search(query, n=5):
    vec = bert_model.encode([query])
    scores = cosine_similarity(vec, bert_embeddings)[0]
    idx = np.argsort(scores)[::-1][:n]
    return df.iloc[idx]

# =====================================================
# WEIGHTED RATING (IMDb STYLE)
# =====================================================
C = df["vote_average"].mean()
m = df["vote_count"].quantile(0.70)

def weighted_rating(x):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v / (v + m)) * R + (m / (v + m)) * C

def top_movies_by_rating(mask=None, n=5):
    data = df.copy()
    if mask is not None:
        data = data[mask]

    data = data[data["vote_count"] >= 30]
    if data.empty:
        return None

    data["score"] = data.apply(weighted_rating, axis=1)
    return data.sort_values(
        by=["score", "vote_count"],
        ascending=False
    ).head(n)

# =====================================================
# GENRE, ANIME, MOOD
# =====================================================
GENRES = [
    "action", "adventure", "animation", "comedy", "crime",
    "drama", "family", "fantasy", "horror", "music",
    "mystery", "romance", "science fiction", "thriller"
]

def extract_genre(text):
    for g in GENRES:
        if g in text:
            return g
    if "anime" in text:
        return "animation"
    return None

def anime_mask():
    return (
        df["genres"].str.lower().str.contains("animation", na=False) &
        (df["original_language"] == "ja")
    )

MOODS = {
    "happy": ["comedy", "family"],
    "sad": ["drama", "romance"],
    "exciting": ["action", "thriller", "adventure"],
    "romantic": ["romance"],
    "scary": ["horror"],

    # Indonesia
    "senang": ["comedy", "family"],
    "sedih": ["drama", "romance"],
    "menegangkan": ["action", "thriller"],
    "romantis": ["romance"],
    "seram": ["horror"]
}

def extract_mood(text):
    for mood in MOODS:
        if mood in text:
            return mood
    return None

# =====================================================
# SIMILAR MOVIE (FUZZY TITLE)
# =====================================================
def extract_similar_title(text):
    for k in ["like", "similar to", "mirip", "seperti"]:
        if k in text:
            return text.split(k, 1)[-1].strip()
    return None

def find_movie_by_title(title, threshold=80):
    titles = df["title"].str.lower().tolist()
    match, score, idx = process.extractOne(
        title.lower(), titles, scorer=fuzz.token_sort_ratio
    )
    if score >= threshold:
        return df.iloc[idx]
    return None

def similar_movies_from_title(title, n=5):
    movie = find_movie_by_title(title)
    if movie is None:
        return None

    idx = movie.name
    vec = bert_embeddings[idx].reshape(1, -1)
    scores = cosine_similarity(vec, bert_embeddings)[0]
    scores[idx] = 0

    top_idx = np.argsort(scores)[::-1][:n]
    return df.iloc[top_idx]

# =====================================================
# MOVIE CARD
# =====================================================
def movie_card(row):
    genres = " • ".join([g.strip() for g in row["genres"].split(",") if g.strip()])
    return f"""
<div style="padding:15px;border-radius:12px;background:#1c1f26;margin-bottom:12px;">
<h4>🎬 {row['title']}</h4>
<p>⭐ <b>{row['vote_average']:.1f}</b> | 👥 {int(row['vote_count'])}</p>
<p>🎭 {genres}</p>
</div>
"""

# =====================================================
# SIDEBAR HELP
# =====================================================
with st.sidebar:
    st.markdown("## 🤖 Help")
    st.markdown("""
    **Try asking:**

    • `saya sedang sedih`
    • `happy movie`
    • `anime comedy`
    • `recommended horror`
    • `random action`
    • `movies like Titanic`
    • `saya ingin mencari movie tentang perang antar hewan`
    """)

# =====================================================
# CHAT STATE
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# =====================================================
# USER INPUT
# =====================================================
user_input = st.chat_input("Ask in Indonesian or English...")

# =====================================================
# CHAT LOGIC (FINAL ORDER)
# =====================================================
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    text = user_input.lower()
    lang = detect_language(text)
    response = ""
    handled = False

    # 1. RECOMMENDED
    if any(k in text for k in ["recommend", "recommended", "rekomendasi"]):
        genre = extract_genre(text)
        mask = df["genres"].str.lower().str.contains(genre, na=False) if genre else None
        movies = top_movies_by_rating(mask)

        response = "⭐ **Top Recommended Movies**\n\n"
        for _, r in movies.iterrows():
            response += movie_card(r)
        handled = True

    # 2. RANDOM
    elif "random" in text or "acak" in text:
        genre = extract_genre(text)
        movies = df[df["genres"].str.lower().str.contains(genre)] if genre else df
        response = "🎲 **Random Movies**\n\n"
        for _, r in movies.sample(min(5, len(movies))).iterrows():
            response += movie_card(r)
        handled = True

    # 3. ANIME
    elif "anime" in text:
        genre = extract_genre(text)
        movies = df[anime_mask()]
        if genre:
            movies = movies[movies["genres"].str.lower().str.contains(genre)]
        response = "🍥 **Anime Movies**\n\n"
        for _, r in movies.sort_values("vote_average", ascending=False).head(5).iterrows():
            response += movie_card(r)
        handled = True

    # 4. MOOD
    elif extract_mood(text):
        mood = extract_mood(text)
        genres = MOODS[mood]
        mask = False
        for g in genres:
            mask |= df["genres"].str.lower().str.contains(g, na=False)
        movies = top_movies_by_rating(mask)

        response = f"🎭 **Movies for '{mood.title()}' mood**\n\n"
        for _, r in movies.iterrows():
            response += movie_card(r)
        handled = True

    # 5. SIMILAR MOVIE
    elif extract_similar_title(text):
        title = extract_similar_title(text)
        movies = similar_movies_from_title(title)

        if movies is None and lang == "id":
            translated = auto_translate(user_input, "id", "en")
            title = extract_similar_title(translated.lower())
            movies = similar_movies_from_title(title)

        if movies is not None:
            response = f"🎬 **Movies similar to {title.title()}**\n\n"
            for _, r in movies.iterrows():
                response += movie_card(r)
        else:
            response = "❌ Movie not found."
        handled = True

    # 6. TF-IDF DESCRIPTIVE SEARCH
    elif len(text.split()) >= 5:
        movies = tfidf_search(user_input)
        response = "🔍 **Movies matching your description**\n\n"
        for _, r in movies.iterrows():
            response += movie_card(r)
        handled = True

    # 7. BERT FALLBACK
    if not handled:
        movies = bert_search(user_input)
        response = "🧠 **Movies matching your request**\n\n"
        for _, r in movies.iterrows():
            response += movie_card(r)

    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})

