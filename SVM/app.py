import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

st.set_page_config(page_title="SVM", layout="centered")
st.title("SVM Classifier: Obese vs Fit")
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

X_full = df[["Height", "Weight"]].to_numpy(dtype=float)
y_full = df["Label"].astype(str).to_numpy()

# ---------- 2. Sidebar controls ----------
st.sidebar.header("Model Settings")
kernel = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"], index=0)
C_value = st.sidebar.slider("C (Regularization)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
if kernel in ("rbf", "poly", "sigmoid"):
    gamma_option = st.sidebar.selectbox("Gamma", ["scale", "auto"], index=0)
else:
    gamma_option = "scale"

test_size = st.sidebar.slider("Test set size", min_value=0.1, max_value=0.4, value=0.2, step=0.05,
                                help="Fraction of the data held out for testing (never seen during training).")

# Speed note: SVM training is O(n^2)-O(n^3), so training on the full set is too slow live.
# Use a stratified sample from the training split for actual training.
train_size = st.sidebar.slider("Training sample size", min_value=200, max_value=3000, value=1000, step=100,
                                 help="SVM is slow on large datasets — a sample keeps the demo responsive.")

st.sidebar.header("Classify a New Person")
input_height = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0)
input_weight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=180.0, value=70.0)
classify_btn = st.sidebar.button("Classify")

# ---------- 3. Split, sample, scale, train ----------
@st.cache_resource
def train_model(kernel, C_value, gamma_option, train_size, test_size, seed=42):
    # First, a genuine train/test split — the test portion is never used for training.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=seed, stratify=y_full
    )

    # From the training portion, draw a stratified sample for speed.
    per_class = train_size // 2
    train_df = pd.DataFrame(X_train_full, columns=["Height", "Weight"])
    train_df["Label"] = y_train_full

    sample_parts = []
    for label in CATEGORY_ORDER:
        group = train_df[train_df["Label"] == label]
        n = min(len(group), per_class)
        sample_parts.append(group.sample(n, random_state=seed))
    sample_df = pd.concat(sample_parts, ignore_index=True)

    X_train = sample_df[["Height", "Weight"]].to_numpy(dtype=float)
    y_train = sample_df["Label"].astype(str).to_numpy()

    # Cap test set size for plotting/evaluation speed (still a genuinely held-out set)
    if len(X_test) > 3000:
        idx = np.random.default_rng(seed).choice(len(X_test), 3000, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel=kernel, C=C_value, gamma=gamma_option, probability=True)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

model, scaler, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = train_model(
    kernel, C_value, gamma_option, train_size, test_size
)

# ---------- 4. Decision boundary plot ----------
def plot_decision_boundary(model, scaler, X_train, y_train, new_point=None):
    h_min, h_max = X_train[:, 0].min() - 5, X_train[:, 0].max() + 5
    w_min, w_max = X_train[:, 1].min() - 5, X_train[:, 1].max() + 5
    xx, yy = np.meshgrid(
        np.linspace(h_min, h_max, 100),
        np.linspace(w_min, w_max, 100)
    )
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler.transform(grid_raw)
    grid_labels = model.predict(grid_scaled)

    label_to_num = {label: i for i, label in enumerate(CATEGORY_ORDER)}
    grid_nums = np.array([label_to_num[l] for l in grid_labels]).reshape(xx.shape)

    cmap = ListedColormap([CATEGORY_COLORS[c] for c in CATEGORY_ORDER])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, grid_nums, alpha=0.3, cmap=cmap, levels=[-0.5, 0.5, 1.5])

    for label in CATEGORY_ORDER:
        m = y_train == label
        if m.sum() == 0:
            continue
        ax.scatter(X_train[m, 0], X_train[m, 1], label=label, c=CATEGORY_COLORS[label],
                   edgecolor="k", s=25, alpha=0.7)

    if new_point is not None:
        ax.scatter(new_point[0], new_point[1], c="black", marker="*", s=350,
                   edgecolor="white", label="New Person", zorder=5)

    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)")
    ax.set_title(f"SVM Decision Boundary (kernel={kernel})")
    ax.legend(loc="upper left", fontsize=8)
    return fig

# ---------- 5. Model evaluation: train vs test accuracy + confusion matrix ----------
st.subheader("Model Evaluation")

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_acc*100:.1f}%", help=f"Evaluated on {len(X_train):,} training samples")
col2.metric("Test Accuracy", f"{test_acc*100:.1f}%", help=f"Evaluated on {len(X_test):,} unseen test samples")

if abs(train_acc - test_acc) > 0.05:
    st.warning("Training and test accuracy differ by more than 5% — this can indicate overfitting.")

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

# ---------- 6. Display: prediction + plot ----------
if classify_btn:
    input_row = [[input_height, input_weight]]
    input_scaled = scaler.transform(input_row)
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
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

    fig = plot_decision_boundary(model, scaler, X_train, y_train, new_point=(input_height, input_weight))
    if fig:
        st.pyplot(fig)
else:
    fig = plot_decision_boundary(model, scaler, X_train, y_train)
    if fig:
        st.pyplot(fig)