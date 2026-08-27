import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD

# 1. Page Configuration
st.set_page_config(
    page_title="Digital Library | Book Recommender",
    page_icon="📚",
    layout="wide"
)

# 2. Hybrid Engine Class
class DigitalLibraryHybridRecommender:
    def __init__(self, books_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.books = books_df.copy()
        self.ratings = ratings_df.copy()
        self.tfidf_matrix = None
        self.indices = None
        self.svd_model = None
        
        self._prepare_content_engine()
        self._prepare_collaborative_engine()

    def _prepare_content_engine(self):
        self.books['metadata_soup'] = (
            self.books['genres'].fillna('') + ' ' +
            self.books['authors'].fillna('') + ' ' +
            self.books['description'].fillna('')
        )
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        self.tfidf_matrix = tfidf.fit_transform(self.books['metadata_soup'])
        self.indices = pd.Series(self.books.index, index=self.books['book_id']).drop_duplicates()

    def _prepare_collaborative_engine(self):
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings[['user_id', 'book_id', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.svd_model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
        self.svd_model.fit(trainset)

    def get_content_recommendations(self, book_id: int, top_n: int = 15) -> pd.DataFrame:
        if book_id not in self.indices:
            return pd.DataFrame()
        idx = self.indices[book_id]
        cosine_sim = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        similar_indices = cosine_sim.argsort()[::-1][1:top_n + 1]
        recs = self.books.iloc[similar_indices].copy()
        recs['content_score'] = cosine_sim[similar_indices]
        return recs

    def recommend_for_user(self, user_id: int, top_k: int = 5, alpha: float = 0.6) -> pd.DataFrame:
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        
        if user_ratings.empty:
            top_books = self.ratings.groupby('book_id')['rating'].agg(['count', 'mean'])
            top_books = top_books[top_books['count'] >= 5].sort_values(by='mean', ascending=False)
            cold_recs = self.books[self.books['book_id'].isin(top_books.index)].head(top_k).copy()
            cold_recs['predicted_rating'] = 4.5
            cold_recs['content_score'] = 0.5
            cold_recs['hybrid_score'] = 0.5
            return cold_recs

        top_user_books = user_ratings.sort_values(by='rating', ascending=False).head(3)['book_id']
        candidate_ids = set()
        
        for b_id in top_user_books:
            content_sims = self.get_content_recommendations(b_id, top_n=10)
            if not content_sims.empty:
                candidate_ids.update(content_sims['book_id'].tolist())

        read_books = set(user_ratings['book_id'].tolist())
        candidate_ids = candidate_ids - read_books
        
        if not candidate_ids:
            candidate_ids = set(self.books['book_id'].tolist()) - read_books

        scored_candidates = []
        for b_id in candidate_ids:
            if b_id not in self.indices:
                continue
            pred_rating = self.svd_model.predict(user_id, b_id).est
            norm_collab_score = (pred_rating - 1.0) / 4.0
            
            idx = self.indices[b_id]
            content_sim_scores = [
                linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix[self.indices[fav_id]])[0][0]
                for fav_id in top_user_books if fav_id in self.indices
            ]
            avg_content_score = np.mean(content_sim_scores) if content_sim_scores else 0.0
            final_score = (alpha * norm_collab_score) + ((1.0 - alpha) * avg_content_score)
            
            scored_candidates.append({
                'book_id': b_id,
                'predicted_rating': pred_rating,
                'content_score': avg_content_score,
                'hybrid_score': final_score
            })

        results_df = pd.DataFrame(scored_candidates).sort_values(by='hybrid_score', ascending=False).head(top_k)
        return results_df.merge(self.books[['book_id', 'title', 'authors', 'genres', 'image_url']], on='book_id', how='left')

# 3. Data & Model Caching
@st.cache_data
def load_data():
    books_url = 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv'
    ratings_url = 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv'
    
    books_raw = pd.read_csv(books_url)
    ratings_raw = pd.read_csv(ratings_url)
    
    books_df = books_raw[['book_id', 'title', 'authors', 'image_url']].copy()
    books_df['genres'] = 'General'
    books_df['description'] = books_df['title'] + ' by ' + books_df['authors']
    
    user_counts = ratings_raw['user_id'].value_counts()
    active_users = user_counts[user_counts >= 15].index[:1200]
    
    ratings_df = ratings_raw[ratings_raw['user_id'].isin(active_users)].copy()
    valid_books = set(books_df['book_id']).intersection(set(ratings_df['book_id']))
    
    books_df = books_df[books_df['book_id'].isin(valid_books)].reset_index(drop=True)
    ratings_df = ratings_df[ratings_df['book_id'].isin(valid_books)].reset_index(drop=True)
    return books_df, ratings_df

@st.cache_resource
def get_model(books_df, ratings_df):
    return DigitalLibraryHybridRecommender(books_df, ratings_df)

# 4. Streamlit Dashboard Layout
st.title("📚 Digital Library Recommender Portal")
st.markdown("### Mitigating Information Overload with Hybrid Filtering")

with st.spinner("Loading library catalog and training latent factors..."):
    books_df, ratings_df = load_data()
    recommender = get_model(books_df, ratings_df)

st.sidebar.header("⚙️ User & Engine Settings")
available_users = sorted(ratings_df['user_id'].unique().tolist())
selected_user_id = st.sidebar.selectbox("Select Student / User ID", options=available_users, index=0)
top_k = st.sidebar.slider("Number of Recommendations (Top-K)", min_value=3, max_value=15, value=5, step=1)
alpha = st.sidebar.slider("Hybrid Balance (α Weight)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)

col_history, col_recs = st.columns([1, 2], gap="large")

with col_history:
    st.subheader("👤 User Reading Profile")
    user_history = ratings_df[ratings_df['user_id'] == selected_user_id].merge(books_df, on='book_id')
    user_history = user_history.sort_values(by='rating', ascending=False)
    st.write(f"Total Books Read/Rated: **{len(user_history)}**")
    
    for _, row in user_history.head(5).iterrows():
        with st.container():
            st.markdown(f"**{row['title']}**")
            st.caption(f"Author(s): {row['authors']} | Given Rating: **⭐ {row['rating']:.1f}/5**")
            st.divider()

with col_recs:
    st.subheader(f"🎯 Top-{top_k} Personalized Recommendations")
    recs_df = recommender.recommend_for_user(user_id=selected_user_id, top_k=top_k, alpha=alpha)
    
    if recs_df.empty:
        st.warning("No recommendations could be generated.")
    else:
        for idx, row in recs_df.reset_index().iterrows():
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    if pd.notna(row.get('image_url')) and str(row.get('image_url')).startswith('http'):
                        st.image(row['image_url'], use_container_width=True)
                    else:
                        st.markdown("📘 *(No Cover)*")
                with c2:
                    st.markdown(f"#### #{idx+1}. {row['title']}")
                    st.markdown(f"**Author(s):** {row['authors']}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Rating", f"⭐ {row['predicted_rating']:.2f}/5")
                    m2.metric("Content Match", f"{row['content_score'] * 100:.1f}%")
                    m3.metric("Hybrid Score", f"{row['hybrid_score']:.3f}")
                st.divider()
