import numpy as np
import pandas as pd

def generate_data(n_samples=20000, seed=42, noise_prob=0.08):
    rng = np.random.default_rng(seed)

    shape = rng.choice([1, -1], size=n_samples, p=[0.85, 0.15])     # mostly round (Apple-like)
    texture = rng.choice([1, -1], size=n_samples, p=[0.85, 0.15])   # mostly smooth
    weight = rng.choice([1, -1], size=n_samples, p=[0.5, 0.5])      # heavy vs light, balanced

    # Weighted score: Weight dominates, Shape/Texture contribute a little
    score = 0.6 * weight + 0.25 * shape + 0.15 * texture
    label = np.where(score >= 0, 1, -1)

    # Add some real-world noise (random label flips), matching the imperfect pattern in the demo data
    flip_mask = rng.random(n_samples) < noise_prob
    label[flip_mask] = -label[flip_mask]

    df = pd.DataFrame({
        "shape": shape,
        "texture": texture,
        "weight": weight,
        "label": label
    })
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("dataset.csv", index=False)
    print(f"dataset.csv created with {len(df)} rows.")
    print(df["label"].value_counts().rename({1: "Apple", -1: "Orange"}))
    print(df.head(10))