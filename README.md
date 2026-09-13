# Machine Learning Lab – 5 Algorithms with GUI (Streamlit)

This project implements 5 core Machine Learning algorithms, each with its own interactive GUI built using **Streamlit** — no terminal input/output required. All experiments use a synthetic Height/Weight dataset to classify a person as **Fit** or **Obese** (except the Perceptron experiment, which classifies fruit into 4 categories based on Hagan Chapter 4).

## Experiments Included

| # | Algorithm | Folder | Task |
|---|-----------|--------|------|
| 1 | K-Nearest Neighbors (KNN) | `KNN/` | Fit vs Obese classification |
| 2 | Support Vector Machine (SVM) | `SVM/` | Fit vs Obese classification |
| 3 | Decision Tree | `Decision Tree/` | Fit vs Obese classification |
| 4 | K-Means Clustering | `K-means-clustering/` | Unsupervised clustering on Height/Weight |
| 5 | Perceptron (Hagan Ch. 4) | `Fruit Classification/` | 4-class fruit classification: Watermelon, Banana, Orange, Apple |

## Tech Stack

- Python 3.11
- [scikit-learn](https://scikit-learn.org/) — ML models (KNN, SVM, Decision Tree, K-Means)
- [Streamlit](https://streamlit.io/) — GUI (runs in browser, no terminal I/O)
- pandas, numpy — data handling
- matplotlib — plots and visualizations

## Project Structure

```
ML Algorithm in Python GUI/
├── dataset.csv                      # Shared dataset for KNN, SVM, Decision Tree
├── generate_dataset.py              # Generates the above dataset (20,000 rows)
├── KNN/
│   └── app.py
├── SVM/
│   └── app.py
├── Decision Tree/
│   └── app.py
├── K-means-clustering/
│   ├── app.py
│   ├── generate_dataset.py          # Generates unlabelled dataset
│   └── dataset_unlabelled.csv
├── Fruit Classification/
│   ├── app.py
│   ├── generate_dataset.py
│   └── dataset.csv
├── venv/                            # Virtual environment (not tracked in git)
├── .gitignore
└── README.md
```



## Setup Instructions

### 1. Clone this repository

```bash
git clone https://github.com/sadmantihan/ML-Algorithm-Python-GUI.git
cd ML-Algorithm-Python-GUI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Activate it:**

- Windows (PowerShell):
```powershell
  venv\Scripts\Activate.ps1
```
- Mac/Linux:
```bash
  source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install scikit-learn numpy pandas matplotlib streamlit
```

### 4. (Optional) Regenerate the datasets

The datasets are already included in this repo, but if you want to regenerate them:

```bash
python generate_dataset.py
cd "K-means-clustering"
python generate_dataset.py
cd ..
cd "Fruit Classification"
python generate_dataset.py
cd ..
```

## How to Run Each Experiment

Run these commands **from the project root folder** (important — some apps load their dataset using a relative path):

```bash
streamlit run KNN/app.py
streamlit run SVM/app.py
streamlit run "Decision Tree/app.py"
streamlit run "K-means-clustering/app.py"
streamlit run "Fruit Classification/app.py"
```

Each command opens a new browser tab with that experiment's GUI. You can stop a running app anytime with `Ctrl+C` in the terminal before running the next one.

## What Each Experiment Demonstrates

- **KNN** — classifies a new point by majority vote of its `k` nearest neighbors. Uses a genuine train/test split (adjustable via sidebar slider) and standardized features, so distance calculations aren't skewed by Height's larger numeric range. Reports training vs. test accuracy and a visual confusion matrix on the held-out test set.
- **SVM** — finds the best separating boundary (hyperplane) between classes. Try switching kernels (`linear` vs `rbf`) to see different boundary shapes. Uses a proper train/test split, with a stratified training sample drawn only from the training portion (for speed) while the test set stays untouched. Reports training vs. test accuracy and a visual confusion matrix.
- **Decision Tree** — learns a set of if/else rules to split the data. The app shows the actual tree diagram, feature importance, a train/test split with an overfitting warning if train/test accuracy diverge, and a visual confusion matrix. Unlike KNN and SVM, no feature scaling is needed since the tree only compares raw threshold values.
- **K-Means Clustering** — unsupervised learning; groups similar points together without ever seeing the true Fit/Obese labels. Includes an Elbow Method plot to help pick the number of clusters, and a comparison table showing how well the discovered clusters align with the true labels (used only for validation after training, not for the clustering itself — hence no train/test split or confusion matrix here, unlike the supervised experiments).
- **Perceptron** — the classic single-layer neural network from Hagan's textbook (Chapter 4), trained with the perceptron learning rule (`w ← w + α·e·p`). Classifies a fruit as **Watermelon**, **Banana**, **Orange**, or **Apple** using Shape, Texture, and Weight (encoded as +1/-1). Since the classic perceptron only handles 2 classes, this experiment trains 4 separate perceptrons in a **one-vs-rest** setup — one per fruit — and picks the fruit whose perceptron is most confident. Weight and bias values are tracked and plotted after every epoch, so the learning process itself (not just the final result) can be visualized.

## Evaluation Approach

The three supervised classification experiments (KNN, SVM, Decision Tree) all follow the same evaluation pattern:
- A genuine **train/test split** (adjustable test set size, default 80/20), so accuracy is measured on data the model never saw during training.
- A **confusion matrix**, shown as a labelled heatmap diagram, along with the individual True Positive / False Negative / False Positive / True Negative counts.
- Training vs. test accuracy comparison, with an overfitting warning where relevant.

K-Means Clustering, being unsupervised, does not use a train/test split or confusion matrix — instead, its discovered clusters are compared against the true labels only after training, purely to validate that the clustering found meaningful structure.

## Notes

- The `venv/` folder is excluded from this repo via `.gitignore` — always create your own using the setup steps above.
- All datasets are synthetically generated (not real-world data) for demonstration purposes.
- This project was built using **pandas 3.0**, which stores text columns using an internal PyArrow-backed format by default. This caused a `TypeError` inside scikit-learn's array indexing (e.g., in `train_test_split`). The fix applied throughout this project is to explicitly convert feature/label columns to plain NumPy arrays after loading (`.to_numpy(dtype=float)` / `.astype(str).to_numpy()`).