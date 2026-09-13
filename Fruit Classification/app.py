import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Perceptron - Fruit Classification", layout="centered")
st.title("Perceptron Classifier: Fruit Recognition")
st.write("Based on Hagan, Chapter 4. Classifies a fruit as **Watermelon**, **Banana**, **Orange**, or **Apple** "
         "using Shape, Texture, and Weight (each encoded as +1 / -1).")

FRUIT_NAMES = ["Watermelon", "Banana", "Orange", "Apple"]
FEATURE_NAMES = ["Shape", "Texture", "Weight"]

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
st.write("Class distribution:")
st.bar_chart(df["label"].value_counts())

X = df[["shape", "texture", "weight"]].to_numpy(dtype=float)
y_raw = df["label"].astype(str).to_numpy()

# ---------- 2. Sidebar: training settings ----------
st.sidebar.header("Perceptron Settings")
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=1.0, value=1.0, step=0.01)
max_epochs = st.sidebar.slider("Max Epochs", min_value=1, max_value=100, value=20)
train_btn = st.sidebar.button("Train Perceptrons")

# ---------- 3. Perceptron learning rule (Hagan-style), with weight/bias history ----------
def hardlim(x):
    return 1 if x >= 0 else -1

def train_perceptron(X, y, learning_rate, max_epochs):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    errors_per_epoch = []

    # Record the weights and bias at the END of every epoch (starting with the initial values)
    w_history = [w.copy()]
    b_history = [b]

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
        w_history.append(w.copy())
        b_history.append(b)
        if total_errors == 0:
            break

    return w, b, errors_per_epoch, np.array(w_history), np.array(b_history)

# ---------- 4. Train one-vs-rest: one perceptron per fruit ----------
@st.cache_resource
def train_all_perceptrons(learning_rate, max_epochs):
    models = {}
    for fruit in FRUIT_NAMES:
        y_binary = np.where(y_raw == fruit, 1, -1).astype(float)
        w, b, errors_per_epoch, w_history, b_history = train_perceptron(X, y_binary, learning_rate, max_epochs)
        models[fruit] = {
            "w": w, "b": b,
            "errors_per_epoch": errors_per_epoch,
            "w_history": w_history,
            "b_history": b_history
        }
    return models

models = train_all_perceptrons(learning_rate, max_epochs)

# ---------- 5. Show training results per fruit ----------
st.subheader("Training Results (One Perceptron per Fruit)")

for fruit in FRUIT_NAMES:
    m = models[fruit]
    converged = m["errors_per_epoch"][-1] == 0
    epochs_used = len(m["errors_per_epoch"])

    with st.expander(f"{fruit} perceptron"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Weight (Shape)", f"{m['w'][0]:.2f}")
        col2.metric("Final Weight (Texture)", f"{m['w'][1]:.2f}")
        col3.metric("Final Weight (Weight)", f"{m['w'][2]:.2f}")
        st.metric("Final Bias", f"{m['b']:.2f}")

        if converged:
            st.success(f"Converged after {epochs_used} epoch(s).")
        else:
            st.warning(f"Did not fully converge within {epochs_used} epoch(s) — "
                       f"{m['errors_per_epoch'][-1]} misclassification(s) remain.")

        # --- Errors per epoch ---
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(1, len(m["errors_per_epoch"]) + 1), m["errors_per_epoch"], marker="o", color="#ef4444")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Misclassifications")
        ax.set_title(f"{fruit} vs. Rest — Learning Curve")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        # --- Weight and bias evolution over epochs ---
        st.write("**How the weights and bias changed during training:**")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        epochs_axis = range(len(m["w_history"]))  # epoch 0 = initial values before training
        for j, fname in enumerate(FEATURE_NAMES):
            ax2.plot(epochs_axis, m["w_history"][:, j], marker="o", markersize=3, label=f"w ({fname})")
        ax2.plot(epochs_axis, m["b_history"], marker="o", markersize=3, label="Bias (b)", linestyle="--", color="black")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Value")
        ax2.set_title(f"{fruit} vs. Rest — Weight & Bias Evolution")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)

        # --- Raw table of weight/bias per epoch (useful for the report) ---
        history_df = pd.DataFrame(
            m["w_history"], columns=[f"w_{fname}" for fname in FEATURE_NAMES]
        )
        history_df["bias"] = m["b_history"]
        history_df.index.name = "Epoch"
        st.dataframe(history_df)

# ---------- 6. Overall multi-class accuracy on training data ----------
def predict_fruit(x, models):
    scores = {}
    for fruit in FRUIT_NAMES:
        m = models[fruit]
        scores[fruit] = np.dot(m["w"], x) + m["b"]
    best_fruit = max(scores, key=scores.get)
    return best_fruit, scores

predictions = []
for i in range(len(X)):
    pred_fruit, _ = predict_fruit(X[i], models)
    predictions.append(pred_fruit)

overall_accuracy = np.mean(np.array(predictions) == y_raw)
st.subheader("Overall Multi-Class Training Accuracy")
st.metric("Accuracy (all 4 fruits combined)", f"{overall_accuracy*100:.1f}%")
st.caption("When multiple perceptrons output +1 for the same input, the fruit with the highest net input "
           "(w·p + b) is chosen as the final prediction.")

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
    predicted_fruit, scores = predict_fruit(p, models)

    st.subheader("Classification Result")
    st.success(f"Predicted fruit: **{predicted_fruit}**")

    st.write("Net input (w·p + b) from each fruit's perceptron:")
    scores_df = pd.Series(scores).sort_values(ascending=False)
    st.bar_chart(scores_df)

    st.write(f"Inputs used: Shape={shape_val}, Texture={texture_val}, Weight={weight_val}")

# ---------- 8. Show learned decision functions ----------
st.subheader("Learned Decision Functions")
for fruit in FRUIT_NAMES:
    m = models[fruit]
    st.latex(
        f"a_{{\\text{{{fruit}}}}} = {m['w'][0]:.2f} \\cdot \\text{{shape}} + "
        f"{m['w'][1]:.2f} \\cdot \\text{{texture}} + {m['w'][2]:.2f} \\cdot \\text{{weight}} + ({m['b']:.2f})"
    )