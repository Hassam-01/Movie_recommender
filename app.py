import streamlit as st
import pickle
import pandas as pd


def recommend(movie):

    movie_index = df[df["title"] == movie].index[0]
    similar_movies = sorted(
        list(enumerate(similarity_matrix[movie_index])),
        key=lambda x: x[1],
        reverse=True,
    )
    movies_rec = [df.iloc[i[0]].title for i in similar_movies[1:6]]

    for i in movies_rec:
        st.write(i)

df = pd.DataFrame.from_dict(pickle.load(open("movies.pkl", "rb")))
similarity_matrix = pickle.load(open("similarity.pkl", "rb"))

st.title("Movie Recommender System")

selected_movie = st.selectbox("Select a movie:", df["title"].values)

if st.button("Recommend"):
    recommend(selected_movie)
