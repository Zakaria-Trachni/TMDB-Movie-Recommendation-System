import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity


st.set_page_config(
    layout="wide",
    page_title="Movie recommender",
    page_icon="🎬"
)

# ========================================= LOADING DATA =========================================

@st.cache_data
def load_pickle(path):
    """Load a pickled file and cache the result."""
    return pickle.load(open(path, 'rb'))


# ========================================= API FUNCTIONS =========================================

@st.cache_data(show_spinner=False)
def fetch_poster(movie_tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_tmdb_id}"
    params = {
        "api_key": "e47a6c1de651bf549c9adeb07ca2cdd5",
        "language": "en-US"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        poster_path = data.get('poster_path')
        
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception as e:
        st.warning(f"Failed to fetch poster: {e}")
    
    return None


# ===================================== RECOMMENDATION FUNCTIONS =====================================

def create_movie_index(movies_df):
    # Create a Series mapping movie titles to their DataFrame indices:
    movies_df = movies_df.reset_index(drop=True)
    return pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()


def compute_overview_similarity(movies_df):
    # Compute cosine similarity based on movie overviews using TF-IDF:
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['overview'])
    return linear_kernel(tfidf_matrix, tfidf_matrix)


def compute_metadata_similarity(movies_df):
    # Compute cosine similarity based on credits, genres, and keywords:
    count_vect = CountVectorizer(stop_words='english')
    count_matrix = count_vect.fit_transform(movies_df['metadata_soup'])
    return cosine_similarity(count_matrix, count_matrix)


def get_recommendations(title, movies_df, cosine_sim, top_n=10):
    indices = create_movie_index(movies_df)
    
    # Get the index of the selected movie
    idx = indices[title]
    
    # Get pairwise similarity scores and sort them
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top N similar movies (excluding the movie itself at index 0)
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    return (
        movies_df['title'].iloc[top_indices].values,
        movies_df['id'].iloc[top_indices].values
    )


# =========================================== UI FUNCTIONS ===========================================

def render_movie_grid(movies_df, num_movies=14, columns_per_row=7):
    """Render a grid of movie posters with titles."""
    rows = num_movies // columns_per_row
    
    for row in range(rows):
        cols = st.columns(columns_per_row)
        start_idx = row * columns_per_row
        end_idx = start_idx + columns_per_row
        
        for col_idx, movie_idx in enumerate(range(start_idx, end_idx)):
            with cols[col_idx]:
                poster_url = fetch_poster(movies_df.iloc[movie_idx]['id'])
                if poster_url:
                    st.image(poster_url)
                
                title = movies_df.iloc[movie_idx]['title']
                st.markdown(
                    f"<p style='text-align: center; font-weight: 500;'>{title}</p>",
                    unsafe_allow_html=True
                )


def render_recommendations(recommended_titles, recommended_ids, movies_per_row=5):
    """Render recommended movies in a grid layout."""
    total_movies = len(recommended_titles)
    rows = (total_movies + movies_per_row - 1) // movies_per_row
    
    for row in range(rows):
        cols = st.columns(movies_per_row)
        start_idx = row * movies_per_row
        end_idx = min(start_idx + movies_per_row, total_movies)
        
        for col_idx, movie_idx in enumerate(range(start_idx, end_idx)):
            with cols[col_idx]:
                poster_url = fetch_poster(recommended_ids[movie_idx])
                if poster_url:
                    st.image(poster_url)
                
                title = recommended_titles[movie_idx]
                st.markdown(
                    f"<p style='text-align: center; font-weight: 500;'>{title}</p>",
                    unsafe_allow_html=True
                )


# ============================================= MAIN =============================================

def main():
    # Load data
    all_movies = load_pickle('models/all_movies.pkl')
    best_rated_movies = load_pickle('models/best_rated_movies.pkl')
    popular_movies = all_movies.sort_values('popularity', ascending=False)
    
    # Page title
    st.markdown(
        "<h1 style='text-align: center'>Movie Recommender System</h1>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 1: Best Rated Movies
    st.subheader("Best rated movies of all time")
    render_movie_grid(best_rated_movies)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Section 2: Most Popular Movies
    st.subheader("Most popular movies of all time")
    render_movie_grid(popular_movies)
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Section 3: Content-Based Recommendations
    st.markdown(
        '<div style="font-size: 1.7rem; font-weight: 600; margin: 0 0 -2rem 0; '
        'line-height: 1.4">Select a movie</div>',
        unsafe_allow_html=True
    )
    
    selected_movie = st.selectbox("", all_movies['title'])
    
    if st.button("Recommend a movie"):
        # Recommendations based on overview
        st.subheader("Filtering based on Movies overview")
        cosine_sim_overview = compute_overview_similarity(all_movies)
        rec_titles_1, rec_ids_1 = get_recommendations(selected_movie, all_movies, cosine_sim_overview)
        render_recommendations(rec_titles_1, rec_ids_1)
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Recommendations based on metadata
        st.subheader("Filtering based on Credits, Genres and Keywords")
        cosine_sim_metadata = compute_metadata_similarity(all_movies)
        rec_titles_2, rec_ids_2 = get_recommendations(selected_movie, all_movies, cosine_sim_metadata)
        render_recommendations(rec_titles_2, rec_ids_2)
        st.markdown("<br><br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
