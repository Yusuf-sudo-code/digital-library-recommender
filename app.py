import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Digital Library | Personalized Portal",
    page_icon="📚",
    layout="wide"
)

# Custom Clean CSS Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #555; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .book-card { padding: 1rem; border-radius: 8px; border: 1px solid #e0e0e0; margin-bottom: 1rem; background-color: #ffffff; }
    .metric-box { background-color: #f8f9fa; padding: 0.5rem; border-radius: 6px; text-align: center; border: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if "user_logged_in" not in st.session_state:
    st.session_state.user_logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "my_ratings" not in st.session_state:
    st.session_state.my_ratings = {}  # {book_id: rating}

# ==========================================
# 3. ENGINE CLASS
# ==========================================
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

    def recommend_for_custom_profile(self, user_ratings_dict: dict, top_k: int = 5, alpha: float = 0.6) -> pd.DataFrame:
        if not user_ratings_dict:
            # Cold-start fallback
            top_books = self.ratings.groupby('book_id')['rating'].agg(['count', 'mean'])
            top_books = top_books[top_books['count'] >= 5].sort_values(by='mean', ascending=False)
            cold_recs = self.books[self.books['book_id'].isin(top_books.index)].head(top_k).copy()
            cold_recs['predicted_rating'] = 4.5
            cold_recs['content_score'] = 0.5
            cold_recs['hybrid_score'] = 0.5
            return cold_recs

        top_user_books = [b_id for b_id, r in sorted(user_ratings_dict.items(), key=lambda x: x[1], reverse=True)[:3]]
        candidate_ids = set()
        
        for b_id in top_user_books:
            content_sims = self.get_content_recommendations(b_id, top_n=10)
            if not content_sims.empty:
                candidate_ids.update(content_sims['book_id'].tolist())

        read_books = set(user_ratings_dict.keys())
        candidate_ids = candidate_ids - read_books
        
        if not candidate_ids:
            candidate_ids = set(self.books['book_id'].tolist()) - read_books

        scored_candidates = []
        for b_id in candidate_ids:
            if b_id not in self.indices:
                continue
            
            # Predict collaborative score using baseline latent parameters
            pred_rating = self.svd_model.predict(uid=999999, iid=b_id).est
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

# ==========================================
# 4. DATA & CACHING
# ==========================================
@st.cache_data
def load_data():
    books_url = 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv'
    ratings_url = 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv'
    books_raw = pd.read_csv(books_url)
    ratings_raw = pd.read_csv(ratings_url)
    
    books_df = books_raw[['book_id', 'title', 'authors', 'image_url']].copy()
    books_df['genres'] = 'Academic / Literature'
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

books_df, ratings_df = load_data()
recommender = get_model(books_df, ratings_df)

# ==========================================
# 5. USER INTERFACE FLOW
# ==========================================

# LOGIN SCREEN
if not st.session_state.user_logged_in:
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<div class='main-header'>🏛️ Digital Library Portal</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>Student Personalized Catalog & Book Discovery System</div>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("#### Patron Sign In")
            name_input = st.text_input("Enter Student ID / Full Name", placeholder="e.g., Yusuf / UABJ-2026-CS")
            submitted = st.form_submit_button("Access Digital Library", use_container_width=True)
            
            if submitted and name_input.strip():
                st.session_state.username = name_input.strip()
                st.session_state.user_logged_in = True
                st.rerun()

# MAIN REPOSITORY & RECOMMENDATION DASHBOARD
else:
    # Top Bar / Profile Bar
    top1, top2 = st.columns([3, 1])
    with top1:
        st.markdown(f"<div class='main-header'>📚 Welcome, {st.session_state.username}</div>", unsafe_allow_html=True)
        st.caption("Personalized Catalog Discovery Engine • Hybrid Information Retrieval Pipeline")
    with top2:
        if st.button("🚪 Log Out / Switch User", use_container_width=True):
            st.session_state.user_logged_in = False
            st.session_state.username = ""
            st.session_state.my_ratings = {}
            st.rerun()
            
    st.divider()

    # Sidebar Controls
    st.sidebar.header("⚙️ Recommender Settings")
    top_k = st.sidebar.slider("Recommendations Count (Top-K)", min_value=3, max_value=12, value=5, step=1)
    alpha = st.sidebar.slider(
        "Hybrid Weight (α Balance)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="α = 0.0 (Pure Content-Based) | α = 0.6 (Optimal Hybrid) | α = 1.0 (Pure Collaborative Filtering)"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📖 Add Books to Profile")
    st.sidebar.write("Select books you have read or are interested in to tune your recommendations:")
    
    book_titles = dict(zip(books_df['title'] + " (by " + books_df['authors'] + ")", books_df['book_id']))
    selected_book_title = st.sidebar.selectbox("Search & Select Book", options=list(book_titles.keys()))
    rating_given = st.sidebar.slider("Your Rating", 1.0, 5.0, 4.0, 0.5)
    
    if st.sidebar.button("➕ Add to My Profile", use_container_width=True):
        b_id = book_titles[selected_book_title]
        st.session_state.my_ratings[b_id] = rating_given
        st.sidebar.success("Added to reading profile!")
        st.rerun()

    # Main Grid: Reading Shelf vs Personalized Feed
    col_profile, col_feed = st.columns([1, 2], gap="large")

    with col_profile:
        st.markdown("### 🗂️ My Reading Shelf")
        if not st.session_state.my_ratings:
            st.info("Your shelf is currently empty. Add 2–3 books from the sidebar to activate personalized hybrid recommendations.")
        else:
            st.write(f"Total Books Rated: **{len(st.session_state.my_ratings)}**")
            for b_id, score in list(st.session_state.my_ratings.items()):
                book_row = books_df[books_df['book_id'] == b_id].iloc[0]
                with st.container():
                    st.markdown(f"**{book_row['title']}**")
                    st.caption(f"Author: {book_row['authors']} | Rating: ⭐ **{score:.1f}/5.0**")
                    if st.button("❌ Remove", key=f"del_{b_id}"):
                        del st.session_state.my_ratings[b_id]
                        st.rerun()
                    st.divider()

    with col_feed:
        st.markdown(f"### 🎯 Recommended For You (Top-{top_k})")
        recs = recommender.recommend_for_custom_profile(st.session_state.my_ratings, top_k=top_k, alpha=alpha)
        
        for idx, row in recs.reset_index().iterrows():
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
