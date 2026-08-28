import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Dataset, Reader, SVD

# ==========================================
# 1. PAGE CONFIGURATION & INSTITUTIONAL STYLING
# ==========================================
st.set_page_config(
    page_title="Institutional Digital Library | Recommender Portal",
    layout="wide"
)

# Clean, Academic CSS Styling
st.markdown("""
<style>
    .main-header { font-size: 2.1rem; font-weight: 700; margin-bottom: 0.1rem; color: #1e293b; }
    .sub-caption { color: #64748b; font-size: 0.95rem; margin-bottom: 1.2rem; }
    .catalog-card { padding: 1.2rem; border-radius: 4px; border: 1px solid #e2e8f0; margin-bottom: 1rem; background-color: #ffffff; }
    .stat-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase; font-weight: 600; }
    .stat-val { font-size: 1.1rem; color: #0f172a; font-weight: 700; }
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
    st.session_state.my_ratings = {}

# ==========================================
# 3. HYBRID RECOMMENDATION ENGINE
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
            # Baseline catalog fallback for initial onboarding
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
# 4. DATA INGESTION & CACHING
# ==========================================
@st.cache_data
def load_data():
    books_url = 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv'
    ratings_url = 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv'
    books_raw = pd.read_csv(books_url)
    ratings_raw = pd.read_csv(ratings_url)
    
    books_df = books_raw[['book_id', 'title', 'authors', 'image_url']].copy()
    books_df['genres'] = 'Academic / General Collection'
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
# 5. USER INTERFACE WORKFLOW
# ==========================================

# LOGIN / AUTHENTICATION INTERFACE
if not st.session_state.user_logged_in:
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown("<div class='main-header'>Digital Library Portal</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-caption'>Personalized Resource Discovery & Information Retrieval System</div>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("##### Patron Sign In")
            name_input = st.text_input("Student ID or Name", placeholder="e.g., UABJ-2026-CS / Yusuf")
            submitted = st.form_submit_button("Access Portal", use_container_width=True)
            
            if submitted and name_input.strip():
                st.session_state.username = name_input.strip()
                st.session_state.user_logged_in = True
                st.rerun()

# MAIN REPOSITORY & RECOMMENDATION DASHBOARD
else:
    # Header Bar
    top1, top2 = st.columns([3, 1])
    with top1:
        st.markdown(f"<div class='main-header'>Repository Account: {st.session_state.username}</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-caption'>Personalized Catalog Discovery Engine | Institutional Hybrid Pipeline</div>", unsafe_allow_html=True)
    with top2:
        if st.button("Log Out", use_container_width=True):
            st.session_state.user_logged_in = False
            st.session_state.username = ""
            st.session_state.my_ratings = {}
            st.rerun()
            
    st.divider()

    # Sidebar: Reading Profile Management
    st.sidebar.markdown("### Reading Profile Setup")
    st.sidebar.caption("Select catalog items to calibrate recommendation scoring:")
    
    book_titles = dict(zip(books_df['title'] + " (by " + books_df['authors'] + ")", books_df['book_id']))
    selected_book_title = st.sidebar.selectbox("Search Catalog", options=list(book_titles.keys()))
    rating_given = st.sidebar.slider("Assigned Rating", 1.0, 5.0, 4.0, 0.5)
    
    if st.sidebar.button("Add to Reading Shelf", use_container_width=True):
        b_id = book_titles[selected_book_title]
        st.session_state.my_ratings[b_id] = rating_given
        st.sidebar.success("Item recorded in profile.")
        st.rerun()

    st.sidebar.markdown("---")

    # Evaluator Mode: Collapsible Model Controls
    with st.sidebar.expander("Model Parameters (Evaluator Mode)", expanded=False):
        st.caption("Inspect and tune hybrid weighting and ranking cutoffs:")
        top_k = st.slider("Top-K Retrieval Cutoff", min_value=3, max_value=12, value=5, step=1)
        alpha = st.slider(
            "Hybrid Weight (alpha)",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="alpha = 0.0 (Pure Content-Based) | alpha = 0.6 (Optimal Hybrid) | alpha = 1.0 (Pure Collaborative SVD)"
        )

    # Main Grid: Reading Shelf vs Ranked Recommendations
    col_profile, col_feed = st.columns([1, 2], gap="large")

    with col_profile:
        st.markdown("### Reading Shelf")
        if not st.session_state.my_ratings:
            st.info("Your shelf is empty. Add 2 to 3 catalog items from the sidebar to activate personalized recommendations.")
        else:
            st.caption(f"Total Books Calibrated: {len(st.session_state.my_ratings)}")
            for b_id, score in list(st.session_state.my_ratings.items()):
                book_row = books_df[books_df['book_id'] == b_id].iloc[0]
                with st.container():
                    st.markdown(f"**{book_row['title']}**")
                    st.caption(f"Author: {book_row['authors']} | Score: {score:.1f} / 5.0")
                    if st.button("Remove Item", key=f"del_{b_id}"):
                        del st.session_state.my_ratings[b_id]
                        st.rerun()
                    st.divider()

    with col_feed:
        st.markdown(f"### Recommended Materials (Top-{top_k})")
        recs = recommender.recommend_for_custom_profile(st.session_state.my_ratings, top_k=top_k, alpha=alpha)
        
        for idx, row in recs.reset_index().iterrows():
            with st.container():
                c1, c2 = st.columns([1, 4])
                with c1:
                    if pd.notna(row.get('image_url')) and str(row.get('image_url')).startswith('http'):
                        st.image(row['image_url'], use_container_width=True)
                    else:
                        st.markdown("[No Cover Available]")
                with c2:
                    st.markdown(f"#### {idx+1}. {row['title']}")
                    st.markdown(f"**Author(s):** {row['authors']}")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted Rating", f"{row['predicted_rating']:.2f} / 5.0")
                    m2.metric("Content Match", f"{row['content_score'] * 100:.1f}%")
                    m3.metric("Hybrid Score", f"{row['hybrid_score']:.3f}")
                st.divider()
