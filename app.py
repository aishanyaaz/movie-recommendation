import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(layout="wide")
st.title("🎬 Movie Recommender")


movies = pd.read_csv("movies.csv")
movies['genres'] = movies['genres'].fillna('').str.replace('|', ' ', regex=False)


vectorizer = TfidfVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genres'])
similarity = cosine_similarity(genre_matrix)


import re

@st.cache_data(show_spinner=False)
def fetch_poster(movie_name):
    api_key = "f68a8535115a5a91aad0becf0243e03e"
    
    # Clean title: remove year like " (1995)"
    cleaned_title = re.sub(r"\s+\(\d{4}\)", "", movie_name).strip()

    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={cleaned_title}"
    try:
        response = requests.get(url).json()
        results = response.get('results')
        if results and results[0].get('poster_path'):
            return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
    except:
        pass

    return "https://via.placeholder.com/300x450?text=No+Image"


# Recommendation logic
def recommend(movie_name, top_n=20):
    try:
        idx = movies[movies['title'].str.contains(movie_name, case=False, na=False)].index[0]
    except IndexError:
        return [], []

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    titles = []
    posters = []

    for i in scores:
        movie = movies.iloc[i[0]]['title']
        titles.append(movie)
        posters.append(fetch_poster(movie))

    return titles, posters

# UI
movie_input = st.text_input("🔍 Enter a movie name", placeholder="e.g., Titanic")

if movie_input:
    with st.spinner("🔄 Fetching recommendations... grab some popcorn 🍿"):
        titles, posters = recommend(movie_input)

    if titles:
        cols = st.columns(5)
        for i in range(len(titles)):
            with cols[i % 5]:
                st.image(posters[i], use_container_width=True)
                st.markdown(f"**{titles[i]}**")
    else:
        st.warning("😕 Movie not found! Try another title.")
