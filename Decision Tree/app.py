import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.set_page_config(page_title="Decision Tree", layout="centered")
st.title("Decision Tree Classifier: Obese vs Fit")
st.write("Classify a person as **Fit** or **Obese** based on Height (cm) and Weight (kg).")

CATEGORY_COLORS = {
    "Fit": "#22c55e",      # green
    "Obese": "#ef4444"     # red
}
CATEGORY_ORDER = ["Fit", "Obese"]

# ---------- 1. Load dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    return df

df = load_data()

with st.expander("View dataset (first 10 rows)"):
    st.dataframe(df.head(10))
st.caption(f"Total rows in dataset: {len(df):,}")

X = df[["Height", "Weight"]].values
y = df["Label"].values

# ---------- 2. Sidebar controls ----------
st.sidebar.header("Model Settings")
max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=15, value=4,
                                help="Deeper trees fit the data more closely but can overfit.")
criterion = st.sidebar.selectbox("Split Criterion", ["gini", "entropy", "log_loss"], index=0)
min_samples_split = st.sidebar.slider("Min Samples to Split", min_value=2, max_value=100, value=20)

st.sidebar.header("Classify a New Person")
input_height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
input_weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=180.0, value=70.0)
classify_btn = st.sidebar.button("Classify")

# ---------- 3. Train model ----------
@st.cache_resource
def train_model(max_depth, criterion, min_samples_split):
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(max_depth, criterion, min_samples_split)

# ---------- 4. Decision boundary plot ----------
def plot_decision_boundary(model, X, y, new_point=None):
    h_min, h_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    w_min, w_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(
        np.linspace(h_min, h_max, 100),
        np.linspace(w_min, w_max, 100)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_labels = model.predict(grid_points)

    label_to_num = {label: i for i, label in enumerate(CATEGORY_ORDER)}
    grid_nums = np.array([label_to_num[l] for l in grid_labels]).reshape(xx.shape)

    cmap = ListedColormap([CATEGORY_COLORS[c] for c in CATEGORY_ORDER])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, grid_nums, alpha=0.3, cmap=cmap, levels=[-0.5, 0.5, 1.5])

    rng = np.random.default_rng(0)
    for label in CATEGORY_ORDER:
        mask = y == label
        idx = np.where(mask)[0]
        if len(idx) > 1000:
            idx = rng.choice(idx, 1000, replace=False)
        ax.scatter(X[idx, 0], X[idx, 1], label=label, c=CATEGORY_COLORS[label],
                   edgecolor="k", s=15, alpha=0.5)

    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], c="black", marker="*", s=350,
                   edgecolor="white", label="New Person", zorder=5)

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title(f"Decision Tree Boundary (depth={max_depth})")
    ax.legend(loc="upper left", fontsize=8)
    return fig

# ---------- 5. Display: prediction ----------
if classify_btn:
    input_row = [[input_height, input_weight]]
    prediction = model.predict(input_row)[0]
    proba = model.predict_proba(input_row)[0]
    classes = model.classes_

    bmi_value = input_weight / ((input_height / 100) ** 2)

    st.subheader("Prediction Result")
    if prediction == "Obese":
        st.error(f"Predicted class: **{prediction}**")
    else:
        st.success(f"Predicted class: **{prediction}**")
    st.write(f"Calculated BMI: **{bmi_value:.2f}**")

    st.write("Class probabilities:")
    st.bar_chart(pd.Series(proba, index=classes))

    fig = plot_decision_boundary(model, X, y, new_point=(input_height, input_weight))
    st.pyplot(fig)
else:
    fig = plot_decision_boundary(model, X, y)
    st.pyplot(fig)

# ---------- 6. Feature importance ----------
st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    "Feature": ["Height", "Weight"],
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))

# ---------- 7. Tree structure visualization ----------
st.subheader("Decision Tree Structure")
show_tree = st.checkbox("Show full tree diagram", value=(max_depth <= 4))
if show_tree:
    fig2, ax2 = plt.subplots(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=["Height", "Weight"],
        class_names=model.classes_,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax2
    )
    st.pyplot(fig2)
else:
    st.info("Tree is large at this depth — check the box above to render it (may be slow/cluttered).")

# ---------- 8. Model accuracy on training data ----------
train_accuracy = model.score(X, y)
st.sidebar.metric("Training Accuracy", f"{train_accuracy*100:.1f}%")