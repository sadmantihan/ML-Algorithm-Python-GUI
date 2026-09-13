import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

st.set_page_config(page_title="Decision Tree - Obese vs Fit", layout="centered")
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "dataset.csv")
    df = pd.read_csv(csv_path)
    return df

df = load_data()

with st.expander("View dataset (first 10 rows)"):
    st.dataframe(df.head(10))
st.caption(f"Total rows in dataset: {len(df):,}")

X = df[["Height", "Weight"]].to_numpy(dtype=float)
y = df["Label"].astype(str).to_numpy()

# ---------- 2. Sidebar controls ----------
st.sidebar.header("Model Settings")
max_depth = st.sidebar.slider("Max Depth", min_value=1, max_value=15, value=4,
                                help="Deeper trees fit the data more closely but can overfit.")
criterion = st.sidebar.selectbox("Split Criterion", ["gini", "entropy", "log_loss"], index=0)
min_samples_split = st.sidebar.slider("Min Samples to Split", min_value=2, max_value=100, value=20)
test_size = st.sidebar.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05,
                                help="Fraction of the data held out for testing (never seen during training).")

st.sidebar.header("Classify a New Person")
input_height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
input_weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=180.0, value=70.0)
classify_btn = st.sidebar.button("Classify")

# ---------- 3. Train/test split + train model ----------
@st.cache_resource
def train_model(max_depth, criterion, min_samples_split, test_size, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion=criterion,
        min_samples_split=min_samples_split,
        random_state=seed
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

model, X_train, X_test, y_train, y_test = train_model(max_depth, criterion, min_samples_split, test_size)

# ---------- 4. Decision boundary plot ----------
def plot_decision_boundary(model, X_train, y_train, new_point=None):
    h_min, h_max = X_train[:, 0].min() - 5, X_train[:, 0].max() + 5
    w_min, w_max = X_train[:, 1].min() - 5, X_train[:, 1].max() + 5
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
        mask = y_train == label
        idx = np.where(mask)[0]
        if len(idx) > 1000:
            idx = rng.choice(idx, 1000, replace=False)
        ax.scatter(X_train[idx, 0], X_train[idx, 1], label=label, c=CATEGORY_COLORS[label],
                   edgecolor="k", s=15, alpha=0.5)

    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], c="black", marker="*", s=350,
                   edgecolor="white", label="New Person", zorder=5)

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title(f"Decision Tree Boundary (depth={max_depth}, trained on {len(X_train):,} samples)")
    ax.legend(loc="upper left", fontsize=8)
    return fig

# ---------- 5. Model evaluation: train vs test accuracy + confusion matrix ----------
st.subheader("Model Evaluation")

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_acc*100:.1f}%", help=f"Evaluated on {len(X_train):,} training samples")
col2.metric("Test Accuracy", f"{test_acc*100:.1f}%", help=f"Evaluated on {len(X_test):,} unseen test samples")

if abs(train_acc - test_acc) > 0.05:
    st.warning("Training and test accuracy differ by more than 5% — this can indicate overfitting. "
               "Try lowering Max Depth or increasing Min Samples to Split.")

st.write("**Confusion Matrix (on the held-out test set):**")
cm = confusion_matrix(y_test, test_pred, labels=CATEGORY_ORDER)

fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CATEGORY_ORDER)
disp.plot(ax=ax_cm, cmap="Blues", colorbar=False, values_format="d")
ax_cm.set_title("Confusion Matrix")
st.pyplot(fig_cm)

tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

col1, col2, col3, col4 = st.columns(4)
col1.metric("True Positive (Fit→Fit)", tp)
col2.metric("False Negative (Fit→Obese)", fn)
col3.metric("False Positive (Obese→Fit)", fp)
col4.metric("True Negative (Obese→Obese)", tn)

# ---------- 6. Display: prediction ----------
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

    fig = plot_decision_boundary(model, X_train, y_train, new_point=(input_height, input_weight))
    st.pyplot(fig)
else:
    fig = plot_decision_boundary(model, X_train, y_train)
    st.pyplot(fig)

# ---------- 7. Feature importance ----------
st.subheader("Feature Importance")
importance_df = pd.DataFrame({
    "Feature": ["Height", "Weight"],
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)
st.bar_chart(importance_df.set_index("Feature"))

# ---------- 8. Tree structure visualization ----------
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