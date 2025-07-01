import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("Restaurant Recommender-21063")

# Load saved model and matrix
@st.cache_data
def load_model():
    user_item_matrix = joblib.load("user_item.pkl")
    svd = joblib.load("svd_model.pkl")
    return user_item_matrix, svd

user_item_matrix, svd = load_model()

# Reconstruct ratings
reconstructed_matrix = svd.transform(user_item_matrix) @ svd.components_

# Select user
usernames = user_item_matrix.index.tolist()
selected_user = st.selectbox("Select a user:", usernames)

if selected_user:
    user_idx = user_item_matrix.index.get_loc(selected_user)
    user_ratings = user_item_matrix.iloc[user_idx]
    user_reconstructed = reconstructed_matrix[user_idx]

    # Recommend only unrated restaurants
    unrated = user_ratings[user_ratings == 0]
    preds = pd.Series(user_reconstructed, index=user_item_matrix.columns)
    recommendations = preds[unrated.index].sort_values(ascending=False).head(5)

    st.subheader("Top Restaurant Recommendations:")
    for i, (resto, score) in enumerate(recommendations.items(), 1):
        st.write(f"{i}. **{resto}** — Predicted rating: {score:.2f}")
