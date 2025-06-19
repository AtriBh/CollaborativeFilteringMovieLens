# ------------------ IMPORTS ------------------
import streamlit as st
import pandas as pd
import pickle
import json
import zipfile
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# ------------------ SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data1")

# ------------------ UTILITIES ------------------
def ensure_unzipped(zip_filename, extract_dir):
    zip_path = os.path.join(DATA_DIR, zip_filename)
    extract_path = os.path.join(DATA_DIR, extract_dir)
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_path)
    return extract_path

def find_file(filename, search_dir):
    for root, _, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {search_dir}")

# ------------------ CACHED LOADERS ------------------
@st.cache_data
def load_movie_titles():
    json_path = os.path.join(DATA_DIR, 'movie_titles.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_pickles():
    ensure_unzipped('user_ratings.zip', 'user_ratings_extract')
    ensure_unzipped('cf_sim_df.zip', 'cf_sim_df_extract')
    user_pkl = find_file('user_ratings.pkl', DATA_DIR)
    cf_pkl = find_file('cf_sim_df.pkl', DATA_DIR)
    with open(user_pkl, 'rb') as f:
        user_ratings = pickle.load(f)
    with open(cf_pkl, 'rb') as f:
        cf_sim_df = pickle.load(f)
    return user_ratings, cf_sim_df

@st.cache_resource
def load_guest_mode_assets():
    ensure_unzipped('features.zip', 'features_extract')
    model_path = os.path.join(DATA_DIR, 'autoencoder_model.keras')
    df_c_path = find_file('df_c.pkl', DATA_DIR)
    genre_path = find_file('genre_columns.pkl', DATA_DIR)
    features_path = find_file('features.pkl', DATA_DIR)
    with open(df_c_path, 'rb') as f:
        df_c = pickle.load(f)
    with open(genre_path, 'rb') as f:
        genre_columns = pickle.load(f)
    with open(features_path, 'rb') as f:
        features = pickle.load(f)
    model = load_model(model_path)
    return df_c, features, genre_columns, model

# ------------------ CF RECOMMENDER ------------------
def get_recommendations(user_id, corrMatrix, userRatings, rating_threshold=4, top_n=20):
    if user_id not in userRatings.index:
        return []
    user_rated = userRatings.loc[user_id].dropna()
    user_rated = user_rated[user_rated > rating_threshold]
    sim_candidates = pd.Series(dtype=float)
    for item, rating in user_rated.items():
        if item in corrMatrix:
            sims = corrMatrix[item].dropna() * rating
            sim_candidates = pd.concat([sim_candidates, sims])
    cf_scores = sim_candidates.groupby(level=0).sum()
    cf_scores = cf_scores.drop(user_rated.index, errors='ignore')
    return cf_scores.nlargest(top_n).index.tolist()

# ------------------ GUEST MODE RECOMMENDER ------------------
def get_guest_recommendations(
    liked_titles, liked_genres, features, df_c, model, genre_columns,
    movie_weight=0.7, genre_weight=0.3, top_n=10
):
    liked_movie_vectors = features[df_c['original_title'].isin(liked_titles)]
    if liked_movie_vectors.empty:
        return pd.DataFrame(columns=['original_title', 'score'])
    liked_movie_mean = liked_movie_vectors.mean()
    genre_vector = pd.Series([0.0] * len(genre_columns), index=genre_columns)
    for g in liked_genres:
        if g in genre_vector:
            genre_vector[g] = 1.0
    genre_vector_full = pd.Series([0.0] * features.shape[1], index=features.columns)
    genre_vector_full.update(genre_vector)
    user_profile_series = movie_weight * liked_movie_mean + genre_weight * genre_vector_full
    user_profile = user_profile_series.values.reshape(1, -1)
    user_embedding = model.predict(user_profile)
    movie_embeddings = model.predict(features.values)
    sim_scores = cosine_similarity(user_embedding, movie_embeddings).flatten()
    df_c = df_c.copy()
    df_c['score'] = sim_scores
    df_c = df_c[~df_c['original_title'].isin(liked_titles)]
    return df_c.sort_values(by='score', ascending=False)[['original_title', 'score']].head(top_n)

# ------------------ STREAMLIT UI ------------------
st.title("🎬 Movie Recommender System")
menu = st.sidebar.radio("Navigation", ["Login", "Guest Mode"])

# ------------------ LOGIN MODE ------------------
if menu == "Login":
    user_ratings, cf_sim_df = load_pickles()
    movie_titles = load_movie_titles()
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    user_input = st.sidebar.text_input("Enter User ID")
    if st.sidebar.button("Login"):
        try:
            uid = int(user_input)
            if uid in user_ratings.index:
                st.session_state.user_id = uid
            else:
                st.warning("User ID not found.")
        except ValueError:
            st.warning("Please enter a numeric User ID.")
    if st.session_state.user_id:
        uid = st.session_state.user_id
        st.subheader(f"Welcome, User {uid}")
        top10 = user_ratings.loc[uid].dropna().sort_values(ascending=False).head(10)
        st.markdown("**Your Top 10 Rated Movies:**")
        st.write(", ".join(top10.index))
        st.markdown("**Recommended for You:**")
        recs = get_recommendations(uid, cf_sim_df, user_ratings)
        for m in recs:
            st.markdown(m)
            rating = st.slider(f"Rate {m}", 0.0, 5.0, 0.0, 0.5, key=m)
            if rating > 0:
                user_ratings.loc[uid, m] = rating
                st.success(f"You rated {m} as {rating}★")
                st.experimental_rerun()

# ------------------ GUEST MODE ------------------
elif menu == "Guest Mode":
    st.subheader("🎟 Guest Mode: Personalized Movie Recommendations")
    df_c, features, genre_columns, model = load_guest_mode_assets()
    available_titles = sorted(df_c['original_title'].dropna().unique().tolist())
    available_genres = sorted(genre_columns)
    liked_titles = st.multiselect("Select Movies You Like", options=available_titles)
    liked_genres = st.multiselect("Select Preferred Genres", options=available_genres)
    if st.button("Get Recommendations"):
        if not liked_titles and not liked_genres:
            st.warning("Please select at least one movie or genre.")
        else:
            results = get_guest_recommendations(
                liked_titles=liked_titles,
                liked_genres=liked_genres,
                features=features,
                df_c=df_c,
                model=model,
                genre_columns=genre_columns,
                top_n=10
            )
            if results.empty:
                st.info("No similar movies found.")
            else:
                st.success("Top 10 Movie Recommendations:")
                st.table(results)

