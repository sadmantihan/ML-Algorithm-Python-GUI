import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="K-Means Clustering", layout="centered")
st.title("K-Means Clustering: Height & Weight Groups")
st.write("Discover natural groupings in unlabelled Height/Weight data — no Fit/Obese labels are used, the algorithm finds structure on its own.")

# ---------- 1. Load unlabelled dataset ----------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset_unlabelled.csv")
    df = pd.read_csv(csv_path)
    return df

df = load_data()

with st.expander("View dataset (first 10 rows)"):
    st.dataframe(df.head(10))
st.caption(f"Total rows in dataset: {len(df):,}")

# ---------- 2. Sidebar controls ----------
st.sidebar.header("Model Settings")
k = st.sidebar.slider("Number of clusters (k)", min_value=2, max_value=8, value=3)

sample_size = st.sidebar.slider("Points to plot", min_value=500, max_value=5000, value=2000, step=500,
                                  help="Plotting fewer points keeps the chart responsive; clustering still runs on the full dataset.")

st.sidebar.header("Classify a New Person")
input_height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
input_weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=180.0, value=70.0)
classify_btn = st.sidebar.button("Assign to Cluster")

# ---------- 3. Prepare features + scale ----------
feature_cols = ["Height", "Weight"]
X = df[feature_cols].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------- 4. Train K-Means ----------
@st.cache_resource
def train_kmeans(X_scaled, k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    return model

model = train_kmeans(X_scaled, k)
df["Cluster"] = model.labels_

cluster_colors = plt.colormaps["tab10"].resampled(k)

# ---------- 5. Elbow method plot ----------
st.subheader("Elbow Method — Choosing k")
st.write("This shows inertia (within-cluster distance) for different k values. The 'elbow' point is often a good choice for k.")

@st.cache_data
def compute_elbow(X_scaled_key_len):
    inertias = []
    k_range = range(1, 9)
    for i in k_range:
        m = KMeans(n_clusters=i, random_state=42, n_init=10)
        m.fit(X_scaled)
        inertias.append(m.inertia_)
    return list(k_range), inertias

k_range, inertias = compute_elbow(len(X_scaled))

fig_elbow, ax_elbow = plt.subplots(figsize=(6, 3.5))
ax_elbow.plot(k_range, inertias, marker="o")
ax_elbow.axvline(x=k, color="red", linestyle="--", label=f"Current k={k}")
ax_elbow.set_xlabel("Number of clusters (k)")
ax_elbow.set_ylabel("Inertia")
ax_elbow.set_title("Elbow Method")
ax_elbow.legend()
st.pyplot(fig_elbow)

# ---------- 6. Cluster scatter plot (Height vs Weight) ----------
def plot_clusters(df, model, new_point_raw=None):
    plot_df = df.sample(min(sample_size, len(df)), random_state=1)

    fig, ax = plt.subplots(figsize=(6, 5))
    for cluster_id in range(k):
        cluster_points = plot_df[plot_df["Cluster"] == cluster_id]
        ax.scatter(cluster_points["Height"], cluster_points["Weight"],
                   label=f"Cluster {cluster_id}", color=cluster_colors(cluster_id),
                   s=15, alpha=0.5, edgecolor="k", linewidth=0.2)

    # Plot centroids (inverse-transformed back to Height/Weight scale)
    centroids_scaled = model.cluster_centers_
    centroids_raw = scaler.inverse_transform(centroids_scaled)
    ax.scatter(centroids_raw[:, 0], centroids_raw[:, 1],
               c="black", marker="X", s=250, edgecolor="white", linewidth=1.5,
               label="Centroids", zorder=5)

    if new_point_raw is not None:
        ax.scatter(new_point_raw[0], new_point_raw[1], c="lime", marker="*", s=400,
                   edgecolor="black", label="New Person", zorder=6)

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title(f"K-Means Clusters (k={k})")
    ax.legend(loc="upper left", fontsize=8)
    return fig

# ---------- 7. Display ----------
if classify_btn:
    input_row = [[input_height, input_weight]]
    input_scaled = scaler.transform(input_row)
    cluster_id = model.predict(input_scaled)[0]

    st.subheader("Cluster Assignment")
    st.success(f"This person belongs to **Cluster {cluster_id}**")

    fig = plot_clusters(df, model, new_point_raw=(input_height, input_weight))
    st.pyplot(fig)
else:
    fig = plot_clusters(df, model)
    st.pyplot(fig)

# ---------- 8. Cluster summary stats ----------
st.subheader("Cluster Summary")
summary = df.groupby("Cluster")[["Height", "Weight"]].mean().round(1)
summary["Count"] = df.groupby("Cluster").size()
st.dataframe(summary)

# ---------- 8.5 Cross-check clusters against true Fit/Obese labels ----------
st.subheader("Cluster vs. True Label Comparison")
st.write("K-means never sees Fit/Obese labels during training. This section checks, "
         "after the fact, how well each discovered cluster aligns with the real BMI-based label.")

# Recompute the true label using the same BMI rule as the labeled dataset
bmi = df["Weight"] / ((df["Height"] / 100) ** 2)
df["True_Label"] = np.where(bmi >= 25, "Obese", "Fit")

cross_tab = pd.crosstab(df["Cluster"], df["True_Label"])
cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

st.write("Counts:")
st.dataframe(cross_tab)

st.write("Percentages (row-wise):")
st.dataframe(cross_tab_pct.round(1))

# Friendly summary sentence per cluster
st.write("**Interpretation:**")
for cluster_id in sorted(df["Cluster"].unique()):
    row = cross_tab_pct.loc[cluster_id]
    dominant_label = row.idxmax()
    dominant_pct = row.max()
    st.write(f"- Cluster {cluster_id} → **{dominant_pct:.1f}% {dominant_label}** "
             f"(looks like a '{dominant_label}' group)")

# ---------- 9. Model metrics ----------
st.sidebar.metric("Inertia (lower = tighter clusters)", f"{model.inertia_:.1f}")