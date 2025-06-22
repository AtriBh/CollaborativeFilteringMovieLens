# CollaborativeFilteringMovieLens


---

### 🔐 Login Note

> You must use a valid **User ID** from the **MovieLens dataset** to log in.
> Specifically, the User ID must be present in the `user_ratings` **DataFrame** used internally by the app.
> Attempting to log in with a random or missing ID will result in an error message.

---

Here’s how that fits in the full `README.md`:

---

```markdown
# 🎬 Movie Recommender System

A real-time interactive movie recommendation web app built using **Streamlit** and the **MovieLens** dataset. This project leverages **Collaborative Filtering** to suggest movies based on a user's past ratings and preferences.

---

## 📌 Features

- 🔐 **User Login** with MovieLens User IDs
- ⭐ View your **Top 10 Rated Movies**
- 🎯 Get **personalized recommendations**
- 🎛️ Rate new movies and update preferences
- 💾 Preprocessed using Pickle for fast loading
- 🚀 Runs entirely in your browser with **Streamlit**

---

## 🧰 Tools & Technologies Used

- **Python** – Core programming
- **Streamlit** – Interactive UI
- **Pandas** – Data manipulation
- **Pickle** – Serialized data loading
- **JSON** – Movie title storage
- **Zipfile / OS** – File handling

---

## 📂 Project Structure

```

📦 movie-recommender
├── 📁 data1/
│   ├── movie\_titles.json
│   ├── user\_ratings.zip → user\_ratings.pkl
│   └── cf\_sim\_df.zip → cf\_sim\_df.pkl
├── app.py
├── requirements.txt
└── README.md

````

---

## 🚶‍♂️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/movie-recommender.git
cd movie-recommender
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## 🔐 Login Note

> You must use a valid **User ID** from the **MovieLens dataset** to log in.
> Specifically, the User ID must exist in the `user_ratings` **DataFrame** used internally by the app.
> Attempting to log in with a random or missing ID will result in an error message.

---

## 🛠️ Steps Involved in Building the Project

1. **Data Preparation**

   * Preprocessed MovieLens data.
   * Saved similarity matrices and user ratings as `.pkl`.

2. **Web App Design**

   * Login page with session state.
   * Dashboard for top-rated and recommended movies.

3. **Recommendation Engine**

   * Based on Collaborative Filtering using correlation matrix.

4. **Interactive Feedback**

   * Users can rate suggested movies to improve future recommendations.

---

## 📌 Future Improvements

* Add **guest mode** with precomputed recommendations.
* Integrate **content-based** or **hybrid filtering**.
* Save updated ratings permanently to a backend (e.g., SQLite, Firebase).
* Add **visuals or posters** using TMDb API.

---

## 📃 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## 🙋‍♂️ Contributing

Pull requests and feedback are welcome. For major changes, please open an issue first to discuss what you'd like to change.

```
