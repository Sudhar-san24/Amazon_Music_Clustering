# # 🎧 Music Cluster Explorer with DBSCAN and K-Means — Styled Dashboard

# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import silhouette_score
# from scipy.stats import zscore
# import matplotlib.pyplot as plt

# # 🎨 Custom Styling
# st.markdown("""
#     <style>
#     .stApp {
#         background-color: #f0f8ff;
#     }
#     .metric-container {
#         background-color: #e6f7ff;
#         padding: 10px;
#         border-radius: 10px;
#         margin-bottom: 10px;
#     }
#     .section-header {
#         font-size: 24px;
#         font-weight: bold;
#         color: #004080;
#         margin-top: 20px;
#     }
#     .dataframe-container {
#         background-color: #ffffff;
#         padding: 10px;
#         border-radius: 10px;
#         box-shadow: 0 0 10px rgba(0,0,0,0.1);
#     }
#     </style>
# """, unsafe_allow_html=True)

# st.set_page_config(page_title="🎧 Music Cluster Explorer", layout="wide")
# st.title("🎧 Music Cluster Explorer")

# # 📁 Upload CSV
# uploaded_file = st.sidebar.file_uploader("Upload your music dataset", type=["csv"])
# if uploaded_file:
#     df = pd.read_csv(uploaded_file)
#     st.success("File uploaded successfully!")

#     # 🎯 Z-score Outlier Removal
#     zscore_cols = [
#         "popularity_songs", "duration_ms", "danceability", "energy",
#         "loudness", "speechiness", "acousticness", "instrumentalness",
#         "liveness", "valence", "tempo", "popularity_artists", "followers"
#     ]
#     z_scores = df[zscore_cols].apply(zscore)
#     df = df[(abs(z_scores) < 3).all(axis=1)]

#     # 🎛️ Feature Selection
#     num_features = zscore_cols
#     cat_features = ["explicit", "mode", "key", "time_signature"]
#     X_num = df[num_features]
#     X_cat = pd.get_dummies(df[cat_features], drop_first=True)
#     X = pd.concat([X_num, X_cat], axis=1)

#     # 🔄 Scaling
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # 🧠 Model Selection
#     model_choice = st.sidebar.selectbox("Choose Clustering Model", ["DBSCAN", "K-Means"])

#     if model_choice == "DBSCAN":
#         eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 0.5, 0.1)
#         min_samples = st.sidebar.slider("min_samples", 1, 20, 5)
#         model = DBSCAN(eps=eps, min_samples=min_samples)
#         labels = model.fit_predict(X_scaled)
#         noise_count = sum(labels == -1)
#         cluster_count = len(set(labels)) - (1 if -1 in labels else 0)

#     elif model_choice == "K-Means":
#         k = st.sidebar.slider("Number of clusters (K)", 2, 20, 5)
#         model = KMeans(n_clusters=k, random_state=42)
#         labels = model.fit_predict(X_scaled)
#         noise_count = "-"
#         cluster_count = k

#     # 🏷️ Assign Cluster Labels
#     df["Cluster"] = labels

#     # 📈 Silhouette Score
#     mask = labels != -1 if model_choice == "DBSCAN" else [True] * len(labels)
#     if sum(mask) > 1:
#         score = silhouette_score(X_scaled[mask], labels[mask])
#         st.markdown('<div class="metric-container">', unsafe_allow_html=True)
#         st.metric("📈 Silhouette Score", round(score, 3))
#         st.markdown('</div>', unsafe_allow_html=True)
#     else:
#         st.warning("Not enough core points to calculate Silhouette Score.")
#         score = None

#     # 📊 PCA Visualization
#     pca = PCA(n_components=2)
#     X_pca = pca.fit_transform(X_scaled)
#     fig, ax = plt.subplots()
#     ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', alpha=0.6)
#     ax.set_title(f"{model_choice} Clusters (PCA 2D)")
#     ax.set_xlabel("PC1")
#     ax.set_ylabel("PC2")
#     st.pyplot(fig)

#     # 📌 Silhouette Score Disclaimer
#     st.markdown("""
#     > **Note:** Silhouette Score is used to evaluate clustering quality.  
#     > For DBSCAN, the score excludes noise points and may vary based on `eps` and `min_samples`.  
#     > A high score (like 0.94) indicates well-separated, dense clusters.
#     """)

#     # 🔍 Dynamic Filters
#     st.sidebar.subheader("🔍 Filter Songs")

#     genre_options = sorted(df["genres"].dropna().unique())
#     artist_options = sorted(df["name_artists"].dropna().unique())

#     selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + genre_options)
#     selected_artist = st.sidebar.selectbox("Select Artist", ["All"] + artist_options)
#     tempo_range = st.sidebar.slider("Tempo range", 50, 200, (80, 140))

#     numeric_filters = {}
#     for col in num_features:
#         min_val = float(df[col].min())
#         max_val = float(df[col].max())
#         selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
#         numeric_filters[col] = selected_range

#     explicit_filter = st.sidebar.selectbox("Explicit", ["All", 0, 1])
#     mode_filter = st.sidebar.selectbox("Mode", ["All", 0, 1])
#     key_filter = st.sidebar.selectbox("Key", ["All"] + list(range(12)))
#     time_signature_filter = st.sidebar.selectbox("Time Signature", ["All", 3, 4, 5, 6, 7])

#     # 🧪 Apply Filters
#     filtered_df = df.copy()
#     if selected_genre != "All":
#         filtered_df = filtered_df[filtered_df["genres"] == selected_genre]
#     if selected_artist != "All":
#         filtered_df = filtered_df[filtered_df["name_artists"] == selected_artist]
#     filtered_df = filtered_df[filtered_df["tempo"].between(*tempo_range)]
#     for col, (min_val, max_val) in numeric_filters.items():
#         filtered_df = filtered_df[filtered_df[col].between(min_val, max_val)]
#     if explicit_filter != "All":
#         filtered_df = filtered_df[filtered_df["explicit"] == explicit_filter]
#     if mode_filter != "All":
#         filtered_df = filtered_df[filtered_df["mode"] == mode_filter]
#     if key_filter != "All":
#         filtered_df = filtered_df[filtered_df["key"] == key_filter]
#     if time_signature_filter != "All":
#         filtered_df = filtered_df[filtered_df["time_signature"] == time_signature_filter]

#     # 📊 Cluster Summary
#     st.markdown('<div class="section-header">📊 Cluster Feature Summary</div>', unsafe_allow_html=True)
#     st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
#     st.write(f"🔢 Number of clusters: {cluster_count}")
#     st.write(f"🧨 Noise points: {noise_count}")
#     st.dataframe(filtered_df.groupby("Cluster")[num_features].mean().round(2))
#     st.markdown('</div>', unsafe_allow_html=True)

#     # 🎵 Song Explorer
#     st.markdown('<div class="section-header">🎵 Song Explorer</div>', unsafe_allow_html=True)
#     st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
#     display_cols = list(dict.fromkeys([
#         "name_song", "name_artists", "genres", "release_date", "explicit", "tempo", "Cluster"
#     ] + num_features))
#     st.dataframe(filtered_df[display_cols])
#     st.markdown('</div>', unsafe_allow_html=True)

#     # 📥 Download
#     st.download_button("Download Clustered Data", filtered_df.to_csv(index=False), "clustered_songs.csv")

# 🎧 Music Cluster Explorer with Evaluation and Interpretation

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import zscore
import matplotlib.pyplot as plt

# 🎨 Custom Styling
st.markdown("""
    <style>
    .stApp { background-color: #f0f8ff; }
    .metric-container {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #004080;
        margin-top: 20px;
    }
    .dataframe-container {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="🎧 Music Cluster Explorer", layout="wide")
st.title("🎧 Music Cluster Explorer")

# 📁 Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload your music dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # 🎯 Z-score Outlier Removal
    zscore_cols = [
        "popularity_songs", "duration_ms", "danceability", "energy",
        "loudness", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "popularity_artists", "followers"
    ]
    z_scores = df[zscore_cols].apply(zscore)
    df = df[(abs(z_scores) < 3).all(axis=1)]

    # 🎛️ Feature Selection
    num_features = zscore_cols
    cat_features = ["explicit", "mode", "key", "time_signature"]
    X_num = df[num_features]
    X_cat = pd.get_dummies(df[cat_features], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)

    # 🔄 Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🧠 Model Selection
    model_choice = st.sidebar.selectbox("Choose Clustering Model", ["DBSCAN", "K-Means"])

    if model_choice == "DBSCAN":
        eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("min_samples", 1, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X_scaled)
        noise_count = sum(labels == -1)
        cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
        inertia = "-"  # Not applicable for DBSCAN

    elif model_choice == "K-Means":
        k = st.sidebar.slider("Number of clusters (K)", 2, 20, 5)
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X_scaled)
        noise_count = "-"
        cluster_count = k
        inertia = model.inertia_

    # 🏷️ Assign Cluster Labels
    df["Cluster"] = labels

    # 📊 Evaluation Metrics
    mask = labels != -1 if model_choice == "DBSCAN" else [True] * len(labels)
    X_eval = X_scaled[mask]
    labels_eval = labels[mask]

    if len(set(labels_eval)) > 1:
        sil_score = silhouette_score(X_eval, labels_eval)
        db_score = davies_bouldin_score(X_eval, labels_eval)
        ch_score = calinski_harabasz_score(X_eval, labels_eval)
    else:
        sil_score = db_score = ch_score = None

    st.markdown('<div class="section-header">📊 Cluster Evaluation Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric("📈 Silhouette Score", round(sil_score, 3) if sil_score else "N/A")
    st.metric("📉 Davies-Bouldin Index", round(db_score, 3) if db_score else "N/A")
    st.metric("📊 Calinski-Harabasz Index", round(ch_score, 3) if ch_score else "N/A")
    st.metric("📦 Inertia (K-Means)", round(inertia, 2) if inertia != "-" else "N/A")
    st.metric("🔢 Number of Clusters", cluster_count)
    st.metric("🧨 Noise Points", noise_count)
    st.markdown('</div>', unsafe_allow_html=True)

    # 📊 PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', alpha=0.6)
    ax.set_title(f"{model_choice} Clusters (PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

    # 🔍 Dynamic Filters
    st.sidebar.subheader("🔍 Filter Songs")
    genre_options = sorted(df["genres"].dropna().unique())
    artist_options = sorted(df["name_artists"].dropna().unique())

    selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + genre_options)
    selected_artist = st.sidebar.selectbox("Select Artist", ["All"] + artist_options)
    tempo_range = st.sidebar.slider("Tempo range", 50, 200, (80, 140))

    numeric_filters = {}
    for col in num_features:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        selected_range = st.sidebar.slider(f"{col} range", min_val, max_val, (min_val, max_val))
        numeric_filters[col] = selected_range

    explicit_filter = st.sidebar.selectbox("Explicit", ["All", 0, 1])
    mode_filter = st.sidebar.selectbox("Mode", ["All", 0, 1])
    key_filter = st.sidebar.selectbox("Key", ["All"] + list(range(12)))
    time_signature_filter = st.sidebar.selectbox("Time Signature", ["All", 3, 4, 5, 6, 7])

    # 🧪 Apply Filters
    filtered_df = df.copy()
    if selected_genre != "All":
        filtered_df = filtered_df[filtered_df["genres"] == selected_genre]
    if selected_artist != "All":
        filtered_df = filtered_df[filtered_df["name_artists"] == selected_artist]
    filtered_df = filtered_df[filtered_df["tempo"].between(*tempo_range)]
    for col, (min_val, max_val) in numeric_filters.items():
        filtered_df = filtered_df[filtered_df[col].between(min_val, max_val)]
    if explicit_filter != "All":
        filtered_df = filtered_df[filtered_df["explicit"] == explicit_filter]
    if mode_filter != "All":
        filtered_df = filtered_df[filtered_df["mode"] == mode_filter]
    if key_filter != "All":
        filtered_df = filtered_df[filtered_df["key"] == key_filter]
    if time_signature_filter != "All":
        filtered_df = filtered_df[filtered_df["time_signature"] == time_signature_filter]

    # 📊 Cluster Feature Summary
    st.markdown('<div class="section-header">📊 Cluster Feature Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(filtered_df.groupby("Cluster")[num_features].mean().round(2))
    st.markdown('</div>', unsafe_allow_html=True)

    # 🧠 Cluster Interpretation
    st.markdown('<div class="section-header">🧠 Cluster Interpretation</div>', unsafe_allow_html=True)
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.write("Use the table below to interpret clusters based on feature profiles. For example:")
    st.markdown("""
    - 🎉 Cluster A: High danceability, energy, valence → **Party tracks**  
    - 🎧 Cluster B: High acousticness, low energy → **Chill acoustic**  
    - 🧘 Cluster C: High instrumentalness, low loudness → **Ambient or instrumental**
    """)
    st.dataframe(filtered_df.groupby("Cluster")[num_features].mean().round(2))
    st.markdown('</div>', unsafe_allow_html=True)

       # 🎵 Song Explorer
    st.markdown('<div class="section-header">🎵 Song Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)

    display_cols = list(dict.fromkeys([
        "name_song", "name_artists", "genres", "release_date", "explicit", "tempo", "Cluster"
    ] + num_features))

    st.dataframe(filtered_df[display_cols])
    st.markdown('</div>', unsafe_allow_html=True)

    # 📥 Download Button
    st.download_button("📥 Download Clustered Data", filtered_df.to_csv(index=False), "clustered_songs.csv")