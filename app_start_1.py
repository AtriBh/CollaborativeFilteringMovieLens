import streamlit as st
import pandas as pd
import pickle
import json
import zipfile
import os

# ------------------ SETUP ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data1")  # directory containing zips and json

# ------------------ SESSION INIT ------------------
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ------------------ FUNCTIONS ------------------

def ensure_unzipped(zip_filename, extract_dir, overwrite=False):
    """
    Unzip zip_filename from DATA_DIR into extract_dir subfolder if needed.
    """
    zip_path = os.path.join(DATA_DIR, zip_filename)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Missing zip file: {zip_path}")
    extract_path = os.path.join(DATA_DIR, extract_dir)
    if not os.path.isdir(extract_path):
        os.makedirs(extract_path, exist_ok=True)
    # unzip into extract_path
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_path)
    return extract_path

@st.cache_data
def load_movie_titles():
    json_path = os.path.join(DATA_DIR, 'movie_titles.json')
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Missing JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Find a file by name under a directory
def find_file(filename, search_dir):
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found in {search_dir}")

@st.cache_resource
def load_pickles():
    # Ensure zips are extracted
    ensure_unzipped('user_ratings.zip', 'user_ratings_extract')
    ensure_unzipped('cf_sim_df.zip',    'cf_sim_df_extract')
    # Find extracted pkl paths
    user_pkl_path = find_file('user_ratings.pkl', DATA_DIR)
    cf_pkl_path   = find_file('cf_sim_df.pkl',    DATA_DIR)
    # Load
    with open(user_pkl_path, 'rb') as f:
        user_ratings = pickle.load(f)
    with open(cf_pkl_path, 'rb') as f:
        cf_sim_df = pickle.load(f)
    return user_ratings, cf_sim_df

# ------------------ LOAD DATA ------------------
user_ratings, cf_sim_df = load_pickles()
movie_titles = load_movie_titles()

# ------------------ RECOMMENDER ------------------
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

# ------------------ APP UI ------------------
# ------------------ APP UI ------------------
st.title("🎬 Movie Recommender System")

if st.session_state.user_id is None:
    st.subheader("Login")
    user_input = st.text_input("Enter User ID")
    if st.button("Login"):
        try:
            uid = int(user_input)
            if uid in user_ratings.index:
                st.session_state.user_id = uid
                st.rerun()  # rerun after login
            else:
                st.warning("User ID not found.")
        except ValueError:
            st.warning("Please enter a numeric User ID.")


# Main content when logged in
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
            st.rerun()
else:
    st.info("Log in with your User ID to see recommendations.")

