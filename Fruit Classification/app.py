import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Perceptron - Apple vs Orange", layout="centered")
st.title("Perceptron Classifier: Apple vs Orange")
st.write("Based on Hagan, Chapter 4. Classifies a fruit as **Apple** or **Orange** using Shape, Texture, and Weight (each encoded as +1 / -1).")

LABEL_MAP = {1: "Apple", -1: "Orange"}

# ---------- 1. Load dataset ----------
@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "dataset.csv")
    df = pd.read_csv(csv_path)
    return df

df = load_data()

with st.expander("View dataset (first 10 rows)"):
    st.dataframe(df.head(10))
st.caption(f"Total rows in dataset: {len(df):,}")

X = df[["shape", "texture", "weight"]].values.astype(float)
y = df["label"].values.astype(float)

# ---------- 2. Sidebar: training settings ----------
st.sidebar.header("Perceptron Settings")
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
max_epochs = st.sidebar.slider("Max Epochs", min_value=1, max_value=100, value=20)
train_btn = st.sidebar.button("Train Perceptron")

# ---------- 3. Perceptron learning rule (Hagan-style) ----------
def hardlim(x):
    return 1 if x >= 0 else -1

def train_perceptron(X, y, learning_rate, max_epochs):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    errors_per_epoch = []

    for epoch in range(max_epochs):
        total_errors = 0
        for i in range(n_samples):
            p = X[i]
            t = y[i]
            a = hardlim(np.dot(w, p) + b)
            e = t - a
            if e != 0:
                w = w + learning_rate * e * p
                b = b + learning_rate * e
                total_errors += 1
        errors_per_epoch.append(total_errors)
        if total_errors == 0:
            break

    return w, b, errors_per_epoch

# ---------- 4. Train (cached so it doesn't retrain on every widget interaction) ----------
@st.cache_resource
def get_trained_model(learning_rate, max_epochs):
    w, b, errors_per_epoch = train_perceptron(X, y, learning_rate, max_epochs)
    return w, b, errors_per_epoch

w, b, errors_per_epoch = get_trained_model(learning_rate, max_epochs)

# ---------- 5. Show training results ----------
st.subheader("Training Result")
col1, col2, col3 = st.columns(3)
col1.metric("Weight (Shape)", f"{w[0]:.2f}")
col2.metric("Weight (Texture)", f"{w[1]:.2f}")
col3.metric("Weight (Weight)", f"{w[2]:.2f}")
st.metric("Bias", f"{b:.2f}")

converged = errors_per_epoch[-1] == 0
epochs_used = len(errors_per_epoch)
if converged:
    st.success(f"Converged after {epochs_used} epoch(s) — no misclassifications on training data.")
else:
    st.warning(f"Did not fully converge within {epochs_used} epoch(s) — "
               f"{errors_per_epoch[-1]} misclassification(s) remain. Try increasing Max Epochs.")

# Training accuracy
predictions = np.array([hardlim(np.dot(w, X[i]) + b) for i in range(len(X))])
train_accuracy = np.mean(predictions == y)
st.metric("Training Accuracy", f"{train_accuracy*100:.1f}%")

# ---------- 6. Convergence plot ----------
st.subheader("Convergence: Errors per Epoch")
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(range(1, len(errors_per_epoch) + 1), errors_per_epoch, marker="o", color="#ef4444")
ax.set_xlabel("Epoch")
ax.set_ylabel("Number of Misclassifications")
ax.set_title("Perceptron Learning Curve")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# ---------- 7. Classify a new fruit ----------
st.sidebar.header("Classify a New Fruit")
shape_input = st.sidebar.selectbox("Shape", ["Round (+1)", "Elongated (-1)"])
texture_input = st.sidebar.selectbox("Texture", ["Smooth (+1)", "Rough (-1)"])
weight_input = st.sidebar.selectbox("Weight", ["Heavy (+1)", "Light (-1)"])
classify_btn = st.sidebar.button("Classify Fruit")

shape_val = 1 if "Round" in shape_input else -1
texture_val = 1 if "Smooth" in texture_input else -1
weight_val = 1 if "Heavy" in weight_input else -1

if classify_btn:
    p = np.array([shape_val, texture_val, weight_val], dtype=float)
    net_input = np.dot(w, p) + b
    prediction = hardlim(net_input)
    predicted_fruit = LABEL_MAP[prediction]

    st.subheader("Classification Result")
    if predicted_fruit == "Apple":
        st.success(f"Predicted fruit: **{predicted_fruit}**")
    else:
        st.error(f"Predicted fruit: **{predicted_fruit}**")

    st.write(f"Net input (w·p + b): **{net_input:.2f}**")
    st.write(f"Inputs used: Shape={shape_val}, Texture={texture_val}, Weight={weight_val}")

# ---------- 8. Show final learned decision rule ----------
st.subheader("Learned Decision Function")
st.latex(
    f"a = \\text{{hardlim}}({w[0]:.2f} \\cdot \\text{{shape}} + {w[1]:.2f} \\cdot \\text{{texture}} + "
    f"{w[2]:.2f} \\cdot \\text{{weight}} + ({b:.2f}))"
)