import numpy as np
import pandas as pd

# Prototype vectors: (shape, texture, weight)
FRUIT_PROTOTYPES = {
    "Watermelon": (1, -1, 1),
    "Banana": (-1, 1, -1),
    "Orange": (1, -1, -1),
    "Apple": (1, 1, -1),
}

def generate_data(n_samples=20000, seed=42, noise_prob=0.08):
    rng = np.random.default_rng(seed)

    fruit_names = list(FRUIT_PROTOTYPES.keys())
    labels = rng.choice(fruit_names, size=n_samples)  # equal probability per fruit

    shape = np.zeros(n_samples, dtype=int)
    texture = np.zeros(n_samples, dtype=int)
    weight = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        s, t, w = FRUIT_PROTOTYPES[labels[i]]
        shape[i] = s
        texture[i] = t
        weight[i] = w

    # Add noise: randomly flip each feature with probability noise_prob
    for arr in (shape, texture, weight):
        flip_mask = rng.random(n_samples) < noise_prob
        arr[flip_mask] = -arr[flip_mask]

    df = pd.DataFrame({
        "shape": shape,
        "texture": texture,
        "weight": weight,
        "label": labels
    })
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("dataset.csv", index=False)
    print(f"dataset.csv created with {len(df)} rows.")
    print(df["label"].value_counts())
    print(df.head(10))