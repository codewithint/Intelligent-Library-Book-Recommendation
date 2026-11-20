# Intelligent Library Book Recommendation System

Built by **Dani Sarkar (`intsarkar`)** | Dept. of CSE (AI & ML), GKCIET

## Overview

Searching for the right reference book in a large library takes a lot of time if you do not already know the exact title or author. I built this content-based book recommendation system using **Python**, **Scikit-Learn**, and **Streamlit** to make discovering related books much faster.

When you pick a book from the catalog, the app checks its **genre, subtopic, author, and description** using TF-IDF vectorization, compares three similarity algorithms, and shows the closest matching books right away.

---

## Features

- **900-Book Dataset across 15 Genres:** Covers 60 books each in Computer Science, AI, Data Science, Cyber Security, Electronics & IoT, Mechanical, Civil, Mathematics, Physics, Biology, Finance, Self-Help, Fiction, Mystery, and World History.
- **3-Algorithm Comparison:** Evaluates **Cosine Similarity**, **K-Nearest Neighbors (KNN)**, and **Manhattan Distance** using `Precision@5`, `Recall@5`, and `F1-Score@5`.
- **Custom Recommendation Count:** Lets you type any custom number of book suggestions you want to see instead of restricting you to fixed dropdown options.
- **EDA & Performance Charts:** Displays the genre distribution (`eda_plot.png`) and model comparison bar chart (`comparison_plot.png`) inside the app.
- **Login & Sign Up:** Includes a working user authentication bar in the top-right corner with session support.
- **Card-Grid Layout:** Shows recommended books in a clean 3-column grid with ratings and similarity match progress bars.

---

## How It Works

```text
900 Books Dataset ──> Text Cleaning & TF-IDF Vectorization
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
Cosine Similarity     K-Nearest Neighbors      Manhattan Distance
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
          Evaluation (Precision@5, Recall@5, F1-Score@5)
                                │
                                ▼
       Streamlit Web App (Auth + Genre Filter + Custom Top-N)
```

---

## Algorithm Comparison

| Algorithm | Metric Used | Ranking Logic | Performance |
| :--- | :--- | :--- | :--- |
| **Cosine Similarity** | Cosine angle between vectors | Higher score (`0` to `1`) = closer match | **1st (Used in live app)** |
| **K-Nearest Neighbors (KNN)** | Euclidean distance | Shorter distance = closer match | **2nd** |
| **Manhattan Distance** | L1 distance | Smaller coordinate difference = closer match | **3rd** |

---

## Project Files

```text
├── h1.py                      # Streamlit web app, login/signup system, and recommendation UI
├── main.py                    # Generates the 900-book dataset, plots, and evaluates all 3 models
├── requirment.txt             # Python dependencies required to run the project
├── library_books_dataset.csv  # Generated dataset of 900 books across 15 genres
├── results.csv                # Saved Precision@5, Recall@5, and F1-Score@5 scores
├── eda_plot.png               # Genre distribution bar chart
├── comparison_plot.png        # Model accuracy comparison chart
├── books_data.pkl             # Saved book metadata
├── similarity_matrix.pkl      # Precomputed Cosine Similarity matrix
└── readme.md                  # Project documentation
```

---

## How to Run Locally

1. **Install the required packages:**
   ```bash
   pip install -r requirment.txt
   ```

2. **Run the training script to generate the dataset, charts, and model files:**
   ```bash
   python main.py 
   ```

3. **Start the Streamlit app:**
   ```bash
   python -m streamlit run h1.py
   ```

---

## Demo Login

You can create a new account using the **Sign Up** button on the top-right corner of the app, or log in directly with the default account:

- **Username:** `dani`
- **Password:** `1234`

---

## Author

- **Name:** Dani Sarkar (`intsarkar`)
- **Course:** B.Tech in Computer Science & Engineering (AI & ML)
- **College:** Ghani Khan Choudhury Institute of Engineering and Technology (GKCIET)
- **Email:** [dansarkar293@gmail.com](mailto:dansarkar293@gmail.com)