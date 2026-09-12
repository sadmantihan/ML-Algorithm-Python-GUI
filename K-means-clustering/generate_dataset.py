import numpy as np
import pandas as pd

def generate_data(n_samples=20000, seed=42):
    rng = np.random.default_rng(seed)

    height = rng.normal(165, 11, n_samples)

    # Weight follows a healthy-BMI-centered distribution with noise
    healthy_weight = 22 * (height / 100) ** 2
    weight = healthy_weight + rng.normal(0, 15, n_samples)
    weight = np.clip(weight, 25, None)  # avoid unrealistic negative/tiny weights

    df = pd.DataFrame({
        "Height": height.round(1),
        "Weight": weight.round(1)
    })
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("dataset_unlabelled.csv", index=False)
    print(f"dataset_unlabelled.csv created with {len(df)} rows.")
    print(df.head())