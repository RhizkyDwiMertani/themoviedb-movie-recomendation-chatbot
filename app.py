import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="🎬 Movie Recommendation Chatbot",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Recommendation Chatbot")
st.caption("Hybrid TF-IDF + BERT | Mood + Genre Aware")

# =====================================================
# SIDEBAR HELP
# =====================================================
with st.sidebar:
    st.markdown("## 🤖 Cara Bertanya")
    st.markdown("""
Gunakan bahasa **Indonesia** atau **Inggris**.  
Kamu bisa bertanya dengan berbagai gaya berikut 👇
""")

    st.markdown("### 🎭 Berdasarkan Mood")
    st.markdown("""
- `saya sedang sedih`
- `aku lagi senang`
- `mood horor`
- `lagi pengen film romantis`
""")

    st.markdown("### 🎬 Berdasarkan Genre")
    st.markdown("""
- `rekomendasi horror`
- `film action terbaik`
- `anime sedih`
- `film thriller`
""")

    st.markdown("### 🔀 Kombinasi Mood + Genre")
    st.markdown("""
- `saya sedang sedih rekomendasikan horror`
- `lagi senang tapi ingin nonton action`
- `anime tapi sedih`
""")

    st.markdown("### 🔍 Deskripsi Bebas (AI Search)")
    st.markdown("""
- `film tentang perang antar hewan`
- `movie about survival in forest`
- `film tentang keluarga yang berpisah`
""")

    st.markdown("### 🧠 Mirip Film Tertentu")
    st.markdown("""
- `movies like Titanic`
- `film mirip Inception`
""")

    st.markdown("---")
    st.caption("🎬 Movie Recommendation Chatbot\n\nHybrid TF-IDF + BERT")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_movies.csv")
    df = df.drop_duplicates(subset="title")

    for col in ["genres", "overview"]:
        df[col] = df[col].fillna("")
    for col in ["vote_count", "vote_average"]:
        df[col] = df[col].fillna(0)

    df["tfidf_text"] = df["title"] + " " + df["overview"] + " " + df["genres"]
    df["bert_text"] = df["overview"] + " " + df["genres"]

    return df.reset_index(drop=True)

df = load_data()

# =====================================================
# TF-IDF
# =====================================================
@st.cache_resource
def build_tfidf(texts):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=15000
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

tfidf_model, tfidf_matrix = build_tfidf(df["tfidf_text"].tolist())

def tfidf_search(query, n=5):
    vec = tfidf_model.transform([query])
    scores = cosine_similarity(vec, tfidf_matrix)[0]
    idx = np.argsort(scores)[::-1][:n]
    return df.iloc[idx]

# =====================================================
# BERT
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
# RATING
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

MOOD_STYLE_HINT = {
    "sad": ["drama", "psychological"],
    "sedih": ["drama", "psychological"]
}

def extract_genre(text):
    for g in GENRES:
        if g in text:
            return g
    if "anime" in text:
        return "animation"
    return None

def extract_mood(text):
    for m in MOODS:
        if m in text:
            return m
    return None

def filter_genre_and_mood(genre, mood):
    data = df[df["genres"].str.lower().str.contains(genre, na=False)]

    if mood in MOOD_STYLE_HINT:
        mask = False
        for tag in MOOD_STYLE_HINT[mood]:
            mask |= data["genres"].str.lower().str.contains(tag, na=False)
        if mask is not False and mask.any():
            data = data[mask]

    if data.empty:
        return None

    data = data[data["vote_count"] >= 30]
    data["score"] = data.apply(weighted_rating, axis=1)
    return data.sort_values("score", ascending=False).head(5)

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
# CHAT
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"], unsafe_allow_html=True)

user_input = st.chat_input("Ask in Indonesian or English...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    text = user_input.lower()
    genre = extract_genre(text)
    mood = extract_mood(text)
    handled = False

    # 1️⃣ GENRE + MOOD
    if genre and mood:
        movies = filter_genre_and_mood(genre, mood)
        if movies is not None:
            response = f"🎬 **{genre.title()} movies for '{mood}' mood**\n\n"
            for _, r in movies.iterrows():
                response += movie_card(r)
        else:
            response = "❌ No movies found."
        handled = True

    # 2️⃣ MOOD ONLY
    elif mood:
        mask = False
        for g in MOODS[mood]:
            mask |= df["genres"].str.lower().str.contains(g, na=False)
        movies = top_movies(mask)
        response = f"🎭 **Movies for '{mood}' mood**\n\n"
        for _, r in movies.iterrows():
            response += movie_card(r)
        handled = True

    # 3️⃣ GENRE ONLY
    elif genre:
        mask = df["genres"].str.lower().str.contains(genre, na=False)
        movies = top_movies(mask)
        response = f"⭐ **Top {genre.title()} movies**\n\n"
        for _, r in movies.iterrows():
            response += movie_card(r)
        handled = True

    # 4️⃣ TF-IDF
    elif len(text.split()) >= 4:
        movies = tfidf_search(user_input)
        response = "🔍 **Movies matching your description**\n\n"
        for _, r in movies.iterrows():
            response += movie_card(r)
        handled = True

    # 5️⃣ BERT
    if not handled:
        movies = bert_search(user_input)
        response = "🧠 **Movies matching your request**\n\n"
        for _, r in movies.iterrows():
            response += movie_card(r)

    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": response})
