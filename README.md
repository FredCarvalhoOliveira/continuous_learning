# Continuous Learning (MovieLens + Streamlit Demo)

An end-to-end playground for incremental learning and lightweight recommendations using the MovieLens ml-latest-small dataset.

- Online learning with River: stream ratings one-by-one, train a logistic regression model on implicit feedback, and visualize a live ROC curve.
- Streamlit UI ("FredFlix"): like movies and get simple content-based recommendations. Optionally enrich movies with poster images from TMDB.

[![Interactive Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fredflix.streamlit.app/)

---

## Contents

- `main.py`: Online learning demo with River and a live ROC plot.
- `demo.py`: Streamlit app to explore movies, like/unlike, and see recommendations.
- `build_data.py`: Optional enrichment script to fetch poster URLs from TMDB.
- `data/`: MovieLens files (`movies.csv`, `ratings.csv`, `links.csv`, `tags.csv`) and dataset README.
- `pyproject.toml`: Project metadata and dependencies.

## Quickstart

### 1) Requirements

- Python 3.12+
- macOS/Linux/Windows

Install dependencies (choose one):

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

The dataset is already included under `data/`.

### 2) (Optional) Enrich with poster images via TMDB

Create a TMDB API key, then set it in a `.env` file at the project root:

```bash
echo "TMDB_API_KEY=your_key_here" > .env
python build_data.py
```

This writes `data/movies_with_posters.csv` and a local `tmdb_cache.json` to avoid repeated API calls.

### 3) Run the online learning demo

```bash
python main.py
```

What it does:
- Loads ratings and movies from `data/`.
- Converts explicit ratings to implicit labels: `implicit = 1` if rating ≥ 4 else 0.
- Features: one-hot of `user_id`, `item_id`, and movie `genres`.
- Model: `OneHotEncoder` → `StandardScaler` → `LogisticRegression` (River pipeline).
- Displays a live ROC curve and AUC updated during streaming.

### 4) Run the Streamlit demo (FredFlix)

```bash
streamlit run demo.py
```

Open the printed local URL in your browser. The app:
- Loads `data/movies_with_posters.csv` if available, otherwise falls back to `data/movies.csv`.
- Lets you filter by genres and like/unlike movies.
- Computes recommendations by building a simple user profile over genres from liked movies and scoring with cosine similarity.

Tips:
- If posters don’t appear, ensure you ran `build_data.py` with a valid TMDB key. The app still works without images.
- Use the sidebar to control the number of top recommendations.

## How it works

### Online learning (`main.py`)
- Streams rows from MovieLens, builds feature dicts per event, calls `model.learn_one(x, y)`.
- Tracks both an online AUC metric (River) and a scikit-learn ROC/AUC computed on the accumulated history for plotting.

### Streamlit recommendations (`demo.py`)
- Preprocesses genres into a binary matrix per movie.
- User profile = normalized sum of liked-movie genre vectors.
- Scores all movies by cosine similarity to the profile, excluding already liked items.

### Data enrichment (`build_data.py`)
- Maps MovieLens `movieId` → `tmdbId` (from `data/links.csv`).
- Calls TMDB to resolve `poster_path` and constructs a `poster_url` column.
- Caches results in `tmdb_cache.json` to be polite and resilient.

## Data and Credits

- MovieLens dataset: see `data/README.md` for license and citation details.
- Posters (optional): data fetched from TMDB (`https://www.themoviedb.org`) under their terms.
