# 🎧 Music Cluster Explorer

## 🔍 Project Overview

Music Cluster Explorer is an interactive Streamlit dashboard designed to analyze and group music tracks based on their audio features using unsupervised machine learning. It empowers users to discover hidden patterns in music data, evaluate clustering performance, and interpret the nature of each cluster—whether it's party anthems, chill acoustic tracks, or ambient instrumentals.

This project blends data science, machine learning, and UI/UX design to deliver a professional-grade tool for music analytics. It’s ideal for data scientists, music analysts, and developers looking to explore clustering techniques in a real-world context.

---

## 🚀 Features

- 📁 Upload your own music dataset (CSV)
- 🧠 Choose clustering algorithm: DBSCAN or K-Means
- 🎛️ Tune hyperparameters interactively
- 📈 View clustering quality metrics:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Index
  - Inertia (K-Means only)
- 📊 Visualize clusters with PCA (2D)
- 🔍 Filter songs by genre, artist, tempo, and feature ranges
- 🧠 Interpret clusters using feature profiles
- 🎵 Explore songs by cluster
- 📥 Download clustered dataset

---

## 📊 Clustering Evaluation Metrics

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| **Silhouette Score** | Measures cohesion vs separation | Close to 1 |
| **Davies-Bouldin Index** | Measures cluster overlap | Lower is better |
| **Calinski-Harabasz Index** | Ratio of between-cluster to within-cluster dispersion | Higher is better |
| **Inertia** | Sum of squared distances to cluster centers (K-Means only) | Lower is better |

These metrics help assess how well the clustering algorithm performed and whether the clusters are meaningful.

---

## 🧠 Cluster Interpretation

After clustering, the dashboard displays the mean values of each feature per cluster. This helps profile and label clusters based on musical characteristics:

- 🎉 High danceability, energy, valence → **Party tracks**
- 🎧 High acousticness, low energy → **Chill acoustic**
- 🧘 High instrumentalness, low loudness → **Ambient or instrumental**

This interpretation helps users understand the nature of each group and apply insights to playlist curation, recommendation systems, or music discovery.

Absolutely, Sudharsan! Here's a polished section you can add to the bottom of your README file, titled **🎼 Involved Features & Musical Insights**. It explains the meaning and relevance of each feature in your dataset, helping users understand how clustering works and why these features matter.

---

## 🎼 Involved Features & Musical Insights

This project uses a rich set of audio and popularity features to cluster songs based on their musical characteristics. Here's what each feature represents:

| Feature | Description |
|--------|-------------|
| **Danceability** | How suitable a track is for dancing based on tempo, rhythm stability, beat strength, and overall regularity |
| **Energy** | Intensity and activity level of a track; high energy often means loud, fast, and noisy |
| **Loudness** | Overall loudness of a track in decibels (dB); helps distinguish soft acoustic songs from loud party tracks |
| **Speechiness** | Presence of spoken words; higher values indicate more speech-like content (e.g., podcasts, rap) |
| **Acousticness** | Confidence measure of whether a track is acoustic; higher values suggest unplugged or natural instrumentation |
| **Instrumentalness** | Predicts whether a track contains vocals; higher values indicate instrumental music |
| **Liveness** | Detects the presence of an audience; higher values suggest live performances or concert recordings |
| **Valence** | Describes the musical positivity conveyed by a track; high valence = happy, cheerful; low valence = sad, moody |
| **Tempo** | Estimated beats per minute (BPM); helps identify fast-paced vs slow tracks |
| **Duration (ms)** | Length of the track in milliseconds |
| **Popularity (songs & artists)** | Spotify popularity score (0–100); reflects how frequently a track or artist is played |
| **Followers** | Number of followers the artist has on Spotify |
| **Explicit** | Indicates whether the track contains explicit content (0 = clean, 1 = explicit) |
| **Mode** | Indicates major (1) or minor (0) scale |
| **Key** | The key of the track (0 = C, 1 = C♯/D♭, ..., 11 = B) |
| **Time Signature** | Number of beats per bar (e.g., 4 = common time, 3 = waltz) |

These features are used to group songs into meaningful clusters, helping users discover patterns like:

- 🎉 High energy + high valence → Party tracks  
- 🎧 High acousticness + low energy → Chill acoustic  
- 🧘 High instrumentalness + low loudness → Ambient or instrumental  

---





# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
