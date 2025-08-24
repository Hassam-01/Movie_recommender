import streamlit as st
import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    movies = pd.read_csv("./Dataset/tmdb_5000_movies.csv")
    credits = pd.read_csv("./Dataset/tmdb_5000_credits.csv")
    movies = movies.merge(credits, on="title")
    return movies

movies = load_data()

# ---------------- Preprocessing ----------------
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

def convertCast(obj):
    L = []
    for i in ast.literal_eval(obj)[:3]:
        L.append(i["name"])
    return L

def convertCrew(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i["job"] == "Director":
            L.append(i["name"])
            break
    return L

@st.cache_data
def preprocess(movies):
    movies = movies[["movie_id", "title", "genres", "keywords", "overview", "cast", "crew"]]
    movies.dropna(inplace=True)

    movies["genres"] = movies["genres"].apply(convert)
    movies["keywords"] = movies["keywords"].apply(convert)
    movies["cast"] = movies["cast"].apply(convertCast)
    movies["crew"] = movies["crew"].apply(convertCrew)

    movies["overview"] = movies["overview"].apply(lambda x: x.split())
    for col in ["genres", "keywords", "cast", "crew", "overview"]:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    df = movies[["movie_id", "title", "tags"]]
    df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())

    # Stemming
    ps = PorterStemmer()
    def stem(text):
        return " ".join([ps.stem(word) for word in text.split()])

    df["tags"] = df["tags"].apply(stem)

    return df

df = preprocess(movies)

# ---------------- Vectorization ----------------
@st.cache_resource
def create_similarity(df):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

similarity = create_similarity(df)

# ---------------- Recommendation Function ----------------
def recommend(movie):
    if movie not in df["title"].values:
        return []

    movie_index = df[df["title"] == movie].index[0]
    distances = similarity[movie_index]
    similar_movies = sorted(
        list(enumerate(distances)), key=lambda x: x[1], reverse=True
    )[1:6]

    return [df.iloc[i[0]].title for i in similar_movies]

# ---------------- Streamlit UI ----------------
st.title("🎬 Movie Recommender System")
st.write("Select a movie and get top 5 similar movies recommended!")

selected_movie_name = st.selectbox("Choose a movie:", df["title"].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie_name)
    if recommendations:
        st.subheader("Recommended Movies:")
        for movie in recommendations:
            st.write("✅", movie)
    else:
        st.error("Sorry! Movie not found in dataset.")
