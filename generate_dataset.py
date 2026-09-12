import numpy as np
import pandas as pd

def generate_data(n_samples=20000, seed=42):
    rng = np.random.default_rng(seed)

    height = rng.normal(165, 11, n_samples)

    # Healthy weight based on BMI ~22, with noise
    healthy_weight = 22 * (height / 100) ** 2
    weight = healthy_weight + rng.normal(0, 8, n_samples)

    bmi = weight / (height / 100) ** 2
    label = np.where(bmi >= 25, "Obese", "Fit")

    df = pd.DataFrame({
        "Height": height.round(1),
        "Weight": weight.round(1),
        "Label": label
    })
    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("dataset.csv", index=False)
    print(f"dataset.csv created with {len(df)} rows.")
    print(df["Label"].value_counts())
    print(df.head())