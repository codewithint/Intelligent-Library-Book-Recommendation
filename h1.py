import streamlit as st
import pandas as pd
import pickle


st.set_page_config(page_title="Smart Library Portal", layout="wide")


st.markdown(
    """
    <style>
    .stAppDeployButton {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
    }
    [data-testid="stMainMenu"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        pointer-events: auto !important;
    }
    div[data-testid="stColumn"]:has(.top-right-auth) {
        position: fixed !important;
        top: 8px !important;
        right: 52px !important;
        width: 280px !important;
        z-index: 999999 !important;
    }
    div[data-testid="stElementContainer"]:has(.top-right-auth) {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


df = pickle.load(open("books_data.pkl", "rb"))
cos_sim = pickle.load(open("similarity_matrix.pkl", "rb"))
results = pd.read_csv("results.csv")


if "users" not in st.session_state:
    st.session_state.users = {"dani": {"password": "1234", "name": "Dani Sarkar"}}
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "auth_box" not in st.session_state:
    st.session_state.auth_box = None


head_left, head_right = st.columns([3.2, 1.3])

with head_left:
    st.title("Intelligent Library Book Recommendation System")
    st.caption("Developed by intsarkar(Dani Sarkar) | Department of CSE (AI & ML)")
    st.caption("email:-  dansarkar293@gmail.com")

with head_right:
    st.markdown('<div class="top-right-auth"></div>', unsafe_allow_html=True)
    if not st.session_state.logged_in:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Login", type="primary", use_container_width=True):
                st.session_state.auth_box = "login"
        with btn_col2:
            if st.button("Sign Up", use_container_width=True):
                st.session_state.auth_box = "signup"
    else:
        with st.container(border=True):
            u_col1, u_col2 = st.columns([2, 1])
            with u_col1:
                st.write(f"**User:** {st.session_state.current_user}")
            with u_col2:
                if st.button("Logout"):
                    st.session_state.logged_in = False
                    st.session_state.current_user = ""
                    st.session_state.auth_box = None
                    st.rerun()


if not st.session_state.logged_in and st.session_state.auth_box == "login":
    _, form_col, _ = st.columns([1, 1.5, 1])
    with form_col:
        with st.container(border=True):
            st.subheader("Student / Member Login")
            u_name = st.text_input("Username:", placeholder="Enter username (e.g. dani)")
            u_pass = st.text_input("Password:", type="password", placeholder="Enter password (e.g. 1234)")
            
            l_col1, l_col2 = st.columns(2)
            with l_col1:
                if st.button("Submit Login", type="primary", use_container_width=True):
                    if u_name in st.session_state.users and st.session_state.users[u_name]["password"] == u_pass:
                        st.session_state.logged_in = True
                        st.session_state.current_user = st.session_state.users[u_name]["name"]
                        st.session_state.auth_box = None
                        st.rerun()
                    else:
                        st.error("Invalid username or password!")
            with l_col2:
                if st.button("Close", use_container_width=True):
                    st.session_state.auth_box = None
                    st.rerun()

elif not st.session_state.logged_in and st.session_state.auth_box == "signup":
    _, form_col, _ = st.columns([1, 1.5, 1])
    with form_col:
        with st.container(border=True):
            st.subheader("Create New Library Account")
            full_name = st.text_input("Full Name:")
            new_user = st.text_input("Choose Username:")
            new_pass = st.text_input("Choose Password:", type="password")
            
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                if st.button("Register Account", type="primary", use_container_width=True):
                    if full_name.strip() == "" or new_user.strip() == "" or new_pass.strip() == "":
                        st.warning("Please fill in all fields.")
                    elif new_user in st.session_state.users:
                        st.error("Username already exists!")
                    else:
                        st.session_state.users[new_user] = {"password": new_pass, "name": full_name}
                        st.success("Account created! Click Login to continue.")
                        st.session_state.auth_box = "login"
                        st.rerun()
            with s_col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.auth_box = None
                    st.rerun()

st.divider()


st.sidebar.header("Library Portal Menu")
page = st.sidebar.radio(
    "Navigate to:",
    ["1. Project Overview", "2. EDA & Model Comparison", "3. Book Recommender (Prototype)"]
)

st.sidebar.divider()
with st.sidebar.container(border=True):
    st.subheader("System Highlights")
    st.write("- **Method:** Content-Based Filtering")
    st.write("- **Vectorizer:** TF-IDF")
    st.write("- **Models:** Cosine, KNN, Manhattan")
    st.write(f"- **Total Books:** {len(df)} Titles")
    st.write(f"- **Categories:** {df['genre'].nunique()} Genres")


if page == "1. Project Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.metric("Total Library Books", f"{len(df)} Books")
    with c2:
        with st.container(border=True):
            st.metric("Subject Categories", f"{df['genre'].nunique()} Genres")
    with c3:
        with st.container(border=True):
            st.metric("Algorithms Tested", "3 Models")
    with c4:
        with st.container(border=True):
            st.metric("Best Algorithm", "Cosine Similarity")

    st.write("")

    left_col, right_col = st.columns(2)

    with left_col:
        with st.container(border=True):
            st.subheader("Problem Statement")
            st.write(
                "In college and public libraries with nearly 1,000 books across 15 departments, "
                "students struggle to find relevant reference books manually. Searching through "
                "large catalogs takes time and requires prior knowledge of specific authors."
            )
            st.info(
                "Goal: Build an intelligent recommendation system that suggests the most relevant "
                "books from a 900-book catalog based on genre, subtopic, author, and description."
            )

    with right_col:
        with st.container(border=True):
            st.subheader("Project Objectives")
            st.write("1. Build and preprocess a dataset of 900 books across 15 diverse genres.")
            st.write("2. Perform Exploratory Data Analysis (EDA) on genre distributions.")
            st.write("3. Extract numerical text features using TF-IDF Vectorizer.")
            st.write("4. Compare 3 algorithms: Cosine Similarity, KNN, and Manhattan Distance.")
            st.write("5. Develop an interactive web prototype for real-time book recommendation.")

    with st.expander(f"View Complete Library Book Catalog ({len(df)} Books)"):
        st.dataframe(df[['book_id', 'title', 'author', 'genre', 'subtopic', 'rating', 'description']], use_container_width=True)


elif page == "2. EDA & Model Comparison":
    with st.container(border=True):
        st.subheader("Performance Evaluation Table (3 ML Algorithms on 900 Books)")
        st.dataframe(results, use_container_width=True)

    st.write("")

    g_col1, g_col2 = st.columns(2)

    with g_col1:
        with st.container(border=True):
            st.subheader("1. Exploratory Data Analysis (15 Genres)")
            st.image("eda_plot.png", use_container_width=True)
            st.caption("Observation: Balanced distribution of 60 books in each of the 15 genres (Total = 900 books).")

    with g_col2:
        with st.container(border=True):
            st.subheader("2. Algorithm Precision Comparison")
            st.image("comparison_plot.png", use_container_width=True)
            st.caption("Observation: Cosine Similarity achieves the highest Precision@5 and F1-Score@5.")


elif page == "3. Book Recommender (Prototype)":
    st.subheader("Discover Similar Books from 900 Titles")

    with st.container(border=True):
        f_col1, f_col2, f_col3 = st.columns([2, 3, 1.5])

        with f_col1:
            genre_list = ["All Genres"] + list(df['genre'].unique())
            selected_genre = st.selectbox("1. Filter by Genre (15 Categories):", genre_list)

        if selected_genre == "All Genres":
            filtered_df = df
        else:
            filtered_df = df[df['genre'] == selected_genre]

        with f_col2:
            selected_book = st.selectbox("2. Select a Book You Like:", filtered_df['title'].values)

        with f_col3:
            
            custom_num_str = st.text_input("3. Show Top (Type Any Number):", value="6", placeholder="e.g. 10, 25, 100")

    selected_row = df[df['title'] == selected_book].iloc[0]

    with st.container(border=True):
        st.markdown(f"#### Selected Book: {selected_row['title']}")
        b_col1, b_col2, b_col3, b_col4 = st.columns(4)
        b_col1.write(f"**Author:** {selected_row['author']}")
        b_col2.write(f"**Genre:** {selected_row['genre']}")
        b_col3.write(f"**Subtopic:** {selected_row['subtopic']}")
        b_col4.write(f"**Rating:** {selected_row['rating']} / 5.0")
        st.write(f"**Summary:** {selected_row['description']}")

    if st.button("Recommend Similar Books", type="primary"):
        if not custom_num_str.strip().isdigit() or int(custom_num_str.strip()) < 1:
            st.warning("Please type a valid number (1 or more) in the 'Show Top' box.")
        else:
            num_rec = int(custom_num_str.strip())
            max_available = len(df) - 1
            if num_rec > max_available:
                num_rec = max_available

            book_index = df[df['title'] == selected_book].index[0]
            sim_scores = list(enumerate(cos_sim[book_index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_rec + 1]

            st.divider()
            st.subheader(f"Top {num_rec} Recommended Books for You")

            cols = st.columns(3)

            for idx_pos, (book_idx, score) in enumerate(sim_scores):
                rec_book = df.iloc[book_idx]
                match_percent = int(round(score * 100))
                if match_percent > 100:
                    match_percent = 100
                elif match_percent < 15:
                    match_percent = 15

                with cols[idx_pos % 3]:
                    with st.container(border=True):
                        st.markdown(f"#### #{idx_pos + 1}. {rec_book['title']}")
                        st.caption(f"By {rec_book['author']} | Subtopic: {rec_book['subtopic']}")
                        st.write(f"**Genre:** {rec_book['genre']}")
                        st.write(f"**Rating:** {rec_book['rating']} / 5.0")
                        st.write(f"**Similarity Match: {round(score * 100, 1)}%**")
                        st.progress(match_percent)
