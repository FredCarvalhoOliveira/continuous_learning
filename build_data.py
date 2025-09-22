import pandas as pd
import requests
import os
import json
import dotenv
from tqdm import tqdm

dotenv.load_dotenv()


TMDB_API_KEY = os.getenv("TMDB_API_KEY")
CACHE_FILE = "tmdb_cache.json"
OUTPUT_FILE = "data/movies_with_posters.csv"

movies = pd.read_csv("data/movies.csv")  # movieId, title, genres

links = pd.read_csv("data/links.csv")  # movieId, imdbId, tmdbId
links = links.dropna(subset=["tmdbId"])
links["tmdbId"] = links["tmdbId"].astype(int)

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        poster_cache = json.load(f)
else:
    poster_cache = {}


def fetch_movie_info(tmdb_id):
    key = str(tmdb_id)
    if key in poster_cache:
        cached_value = poster_cache[key]
        if isinstance(cached_value, dict):
            return cached_value.get("poster_url"), None
        else:
            return cached_value, None

    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        data = requests.get(url, timeout=15).json()
        poster_path = data.get("poster_path")
        poster_url = (
            f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None
        )
        poster_cache[key] = poster_url
        return poster_url, None
    except Exception as e:
        print(f"Error fetching tmdbId {tmdb_id}: {e}")
        poster_cache[key] = None
        return None, None


movie_id_to_tmdb = links.set_index("movieId")["tmdbId"].to_dict()

poster_urls = []
for idx, movie_id in enumerate(
    tqdm(movies["movieId"], desc="Fetching poster URLs"), start=1
):
    tmdb_id = movie_id_to_tmdb.get(movie_id)
    if tmdb_id:
        poster_url, _title = fetch_movie_info(tmdb_id)
    else:
        poster_url = None
    poster_urls.append(poster_url)
    if idx % 10 == 0:
        with open(CACHE_FILE, "w") as f:
            json.dump(poster_cache, f)

movies["poster_url"] = poster_urls

with open(CACHE_FILE, "w") as f:
    json.dump(poster_cache, f)

movies.to_csv(OUTPUT_FILE, index=False)
print(f"Saved enriched movies to {OUTPUT_FILE}")
