import streamlit as st
import pandas as pd
import pickle
import json
import zipfile
import os

# ------------------ PATH SETUP ------------------
DATA_DIR = "data1"
USER_ZIP = os.path.join(DATA_DIR, "user_ratings.zip")
CF_ZIP = os.path.join(DATA_DIR, "cf_sim_df.zip")
USER_PKL = os.path.join(DATA_DIR, "user_ratings.pkl")
CF_PKL = os.path.join(DATA_DIR, "cf_sim_df.pkl")
MOVIE_JSON = os.path.join(DATA_DIR, "movie_titles.json")

# ------------------ UTIL: Unzip If Needed ------------------
def unzip_if_needed(zip_path, target_pkl):
    if not os.path.exists(target_pkl):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)

# ------------------ LOAD DATA ------------------
@st.cache_resource
def load_data():
    unzip_if_needed(USER_ZIP, USER_PKL)
    unzip_if_needed(CF_ZIP, CF_PKL)

    with open(USER_PKL, "rb") as f:
        user_ratings = pickle.load(f)
    with open(CF_PKL, "rb") as f:
        cf_sim_df = pickle.load(f)
    with open(MOVIE_JSON, "r", encoding="utf-8") as f:
        movie_titles = json.load(f)
    return user_ratings, cf_sim_df, movie_titles

user_ratings, cf_sim_df, movie_titles = load_data()

# ------------------ RECOMMENDER ------------------
def get_recommendations(user_id, corrMatrix, userRatings, rating_threshold, top_n=20):
    if user_id not in userRatings.index:
        return []
    user_rated = userRatings.loc[user_id].dropna()
    user_rated = user_rated[user_rated > rating_threshold]

    sim_candidates = pd.Series(dtype=float)
    for item, rating in user_rated.items():
        if item in corrMatrix:
            sims = corrMatrix[item].dropna().map(lambda x: x * rating)
            sim_candidates = pd.concat([sim_candidates, sims])

    cf_scores = sim_candidates.groupby(sim_candidates.index).sum()
    cf_scores = cf_scores.drop(user_rated.index, errors='ignore')
    return cf_scores.sort_values(ascending=False).head(top_n).index.tolist()

# ------------------ UI ------------------
st.title("🎬 Movie Recommender System")

if "user_id" not in st.session_state:
    st.session_state.user_id = None

# Login
user_id_input = st.sidebar.text_input("Enter User ID")
if st.sidebar.button("Login"):
    try:
        user_id = int(user_id_input)
        if user_id in user_ratings.index:
            st.session_state.user_id = user_id
        else:
            st.warning("User ID not found.")
    except:
        st.warning("Invalid ID.")

# ------------------ DASHBOARD ------------------
if st.session_state.user_id is not None:
    st.subheader(f"🎟️ Welcome User {st.session_state.user_id}")
    rated_movies = user_ratings.loc[st.session_state.user_id].dropna().sort_values(ascending=False).head(10)
    st.markdown("### 👍 Your Top 10 Rated Movies")
    st.write(", ".join(rated_movies.index.tolist()))

    st.markdown("### 🎯 Recommended For You")
    recs = get_recommendations(st.session_state.user_id, cf_sim_df, user_ratings, rating_threshold=4, top_n=20)
    for movie in recs:
        st.markdown(f"**{movie}**")
        new_rating = st.slider(f"Rate '{movie}'", 0.0, 5.0, 0.0, 0.5, key=movie)
        if new_rating > 0:
            user_ratings.loc[st.session_state.user_id, movie] = new_rating
            st.success(f"Rated {movie} as {new_rating}★")
            st.experimental_rerun()
else:
    st.info("Please enter your User ID to get started.")
