# ------------------------------------------
# Clean Header (No Emojis)
# ------------------------------------------
st.markdown("<div class='main-header'>Digital Library Portal</div>", unsafe_allow_html=True)
st.caption("Personalized Catalog Discovery Engine • University Institutional Repository")

# ------------------------------------------
# Clean Sidebar with Collapsible Evaluator Mode
# ------------------------------------------
st.sidebar.markdown("### Reading Profile Setup")
st.sidebar.write("Select catalog items to calibrate recommendations:")

book_titles = dict(zip(books_df['title'] + " (by " + books_df['authors'] + ")", books_df['book_id']))
selected_book_title = st.sidebar.selectbox("Catalog Search", options=list(book_titles.keys()))
rating_given = st.sidebar.slider("Assigned Rating", 1.0, 5.0, 4.0, 0.5)

if st.sidebar.button("Add to Shelf", use_container_width=True):
    b_id = book_titles[selected_book_title]
    st.session_state.my_ratings[b_id] = rating_given
    st.sidebar.success("Item added to reading shelf.")
    st.rerun()

st.sidebar.markdown("---")

# Technical Model Parameters Collapsed by Default
with st.sidebar.expander("Model Parameters (Evaluator Mode)", expanded=False):
    st.caption("Fine-tune hybrid weighting & retrieval cutoffs:")
    top_k = st.slider("Top-K Cutoff", min_value=3, max_value=12, value=5, step=1)
    alpha = st.slider(
        "Hybrid Weight (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="0.0 = Pure Content-Based (TF-IDF) | 1.0 = Pure Collaborative (SVD)"
    )
