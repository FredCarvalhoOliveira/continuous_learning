import os
import html
import numpy as np
import pandas as pd
import streamlit as st

CSV_PATH = os.path.abspath("data/movies_with_posters.csv")

st.set_page_config(page_title="FredFlix", layout="wide")


def render_curved_title(text: str) -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Anton:wght@400&display=swap');
        .fredflix-title {
            font-family: 'Anton', sans-serif;
            color: #e50914;
            font-size: 3rem;
            line-height: 1.1;
            margin: 0.5rem 0 1rem 0;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            display: inline-flex;
            align-items: flex-start;
            gap: 0.05em;
        }
        .fredflix-letter {
            display: inline-block;
            transform-origin: top center;
        }
        .fredflix-space { width: 0.5em; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    letters = list(text)
    n = len(letters)
    parts: list[str] = []
    for i, ch in enumerate(letters):
        if ch == " ":
            parts.append('<span class="fredflix-letter fredflix-space">&nbsp;</span>')
            continue
        if n > 1:
            t = abs(i - (n - 1) / 2) / ((n - 1) / 2)
        else:
            t = 1.0
        scale = 1.0 + 0.35 * t
        parts.append(
            f'<span class="fredflix-letter" style="transform: scale({scale});">{html.escape(ch)}</span>'
        )
    html_block = f'<div class="fredflix-title">{"".join(parts)}</div>'
    st.markdown(html_block, unsafe_allow_html=True)


render_curved_title("FredFlix")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.stop()
    df = pd.read_csv(path)
    if "item_id" not in df.columns:
        if "movieId" in df.columns:
            df = df.rename(columns={"movieId": "item_id"})
        elif "itemId" in df.columns:
            df = df.rename(columns={"itemId": "item_id"})
        else:
            df["item_id"] = np.arange(1, len(df) + 1)
    if "title" not in df.columns:
        df["title"] = df["item_id"].apply(lambda x: f"Item #{x}")
    else:
        df["title"] = df["title"].fillna(df["item_id"].apply(lambda x: f"Item #{x}"))
    if "genres" not in df.columns:
        df["genres"] = ""
    df.sort_values(["item_id"], ascending=[True], inplace=True)
    df = df.drop_duplicates(subset=["item_id"], keep="first")
    df["item_id"] = df["item_id"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_genres(movies_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    genres_series = movies_df.get("genres", pd.Series(dtype=str)).fillna("")
    dummies = genres_series.str.get_dummies(sep="|")
    if "(no genres listed)" in dummies.columns:
        dummies = dummies.drop(columns=["(no genres listed)"])
    genre_names = sorted(dummies.columns.tolist())
    dummies = dummies.reindex(columns=genre_names, fill_value=0)
    genres_df = pd.concat(
        [
            movies_df[["item_id"]].astype(int).reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    return genres_df, genre_names


def show_poster(poster_url: str, title: str) -> None:
    if pd.notna(poster_url) and isinstance(poster_url, str) and poster_url.strip():
        safe_url = html.escape(poster_url, quote=True)
        safe_title = html.escape(str(title) if title is not None else "", quote=True)
        st.markdown(
            f'<img src="{safe_url}" alt="{safe_title}" loading="lazy" '
            'style="width:100%;height:auto;border-radius:4px;" />',
            unsafe_allow_html=True,
        )


@st.cache_data(show_spinner=False)
def compute_recommendations(
    movies_df: pd.DataFrame,
    genres_df: pd.DataFrame,
    genre_names: list[str],
    liked_ids: set[int],
    k: int,
) -> pd.DataFrame:
    if not liked_ids:
        return pd.DataFrame(columns=movies_df.columns)

    merged = movies_df.merge(genres_df, on="item_id", how="left")
    liked_mask = merged["item_id"].isin(liked_ids)
    if not liked_mask.any():
        return pd.DataFrame(columns=movies_df.columns)

    G = merged.loc[:, genre_names].fillna(0).to_numpy(dtype=float)
    liked_G = G[liked_mask.values]
    profile = liked_G.sum(axis=0)

    p_norm = np.linalg.norm(profile)
    if p_norm == 0:
        return pd.DataFrame(columns=movies_df.columns)
    profile = profile / p_norm

    movie_norms = np.linalg.norm(G, axis=1)
    denom = movie_norms
    denom[denom == 0] = 1.0
    sims = (G @ profile) / denom

    sims[liked_mask.values] = -np.inf

    merged = merged.assign(similarity=sims)
    recs = (
        merged.sort_values("similarity", ascending=False)
        .head(k)
        .loc[:, ["item_id", "title", "poster_url", "similarity"]]
    )
    return recs


movies = load_data(CSV_PATH)
genres_df, genre_names = load_genres(movies)

if "likes_map" not in st.session_state:
    st.session_state.likes_map = {}

likes_map = st.session_state.likes_map
liked_ids = {int(k) for k in likes_map.keys()}

st.sidebar.title("Controls")
num_recs = st.sidebar.slider("# of top bar recs", 3, 12, 8)

st.markdown("### Top recommendations")
recs = compute_recommendations(movies, genres_df, genre_names, liked_ids, num_recs)
if recs.empty:
    st.info("Like a few movies to see personalized recommendations.")
else:
    cols = st.columns(len(recs))
    for col, (_, row) in zip(cols, recs.iterrows()):
        with col:
            poster_url = row.get("poster_url")
            show_poster(poster_url, row["title"])
            st.caption(row["title"])

st.divider()

left, right = st.columns([3, 1], gap="large")

with right:
    st.markdown("### Your likes")
    likes_container = st.container(height=520, border=True)
    with likes_container:
        liked_movies = movies[movies["item_id"].isin(liked_ids)]
        if liked_movies.empty:
            st.caption("No likes yet. Click Like on a movie below.")
        else:
            for _, row in liked_movies.iterrows():
                poster_url = row.get("poster_url")
                item_id = int(row["item_id"])
                show_poster(poster_url, row["title"])
                cols_like = st.columns([5, 2])
                with cols_like[0]:
                    st.caption(row["title"])
                with cols_like[1]:
                    if st.button(
                        "✕",
                        key=f"unlike_sidebar_{item_id}",
                        help="Remove",
                        use_container_width=True,
                    ):
                        st.session_state.likes_map.pop(str(item_id), None)
                        st.rerun()

with left:
    st.markdown("### All movies")
    filter_col, page_col = st.columns([3, 1], gap="small")
    with filter_col:
        selected_genres = st.multiselect(
            "Filter by genre",
            options=genre_names,
            default=[],
            help="Show only movies matching any selected genres",
        )

    grid_movies = movies
    if selected_genres:
        try:
            mask = genres_df[selected_genres].sum(axis=1) > 0
            allowed_ids = set(genres_df.loc[mask, "item_id"].astype(int))
            grid_movies = movies[movies["item_id"].isin(allowed_ids)]
        except Exception:
            grid_movies = movies

    total_items = len(grid_movies)
    page_size = 40
    total_pages = max(1, (total_items + page_size - 1) // page_size)
    with page_col:
        page_index = st.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            help="Jump to page",
        )

    # Manage page state
    if "grid_page" not in st.session_state:
        st.session_state.grid_page = 1
    if page_index != st.session_state.grid_page:
        st.session_state.grid_page = int(page_index)

    current_page = st.session_state.grid_page
    start = (current_page - 1) * page_size
    end = min(start + page_size, total_items)
    page_movies = grid_movies.iloc[start:end]

    st.caption(f"Showing {start + 1}–{end} of {total_items}")

    grid_container = st.container(height=520, border=True)
    with grid_container:
        cols_count = 5
        grid_cols = st.columns(cols_count)
        i = 0
        for _, row in page_movies.iterrows():
            item_id = int(row["item_id"])
            liked = st.session_state.likes_map.get(str(item_id), False)
            with grid_cols[i % cols_count]:
                poster_url = row.get("poster_url")
                show_poster(poster_url, row["title"])
                st.caption(row["title"])
                btn_label = "Unlike" if liked else "Like"
                if st.button(btn_label, key=f"like_btn_{item_id}"):
                    if liked:
                        st.session_state.likes_map.pop(str(item_id), None)
                    else:
                        st.session_state.likes_map[str(item_id)] = True
                    st.rerun()
            i += 1

st.caption("UI demo powered by data/movies.csv")
