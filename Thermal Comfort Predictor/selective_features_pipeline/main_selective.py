# selective_features_pipeline/main_selective.py

import pandas as pd
from utils.config import TARGET_COL, TOP_K_FEATURES, N_SPLITS
from models.meta_model import train_meta_model

def load_data(filepath):
    """
    Load dataset from Excel file and separate features and target.
    """
    df = pd.read_excel(filepath)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y

def main():
    print("🚀 Loading dataset...")
    X, y = load_data("../dataset/input_dataset.xlsx")  # Go up one directory to access dataset

    print(f"\n🔍 Total features available: {list(X.columns)}")
    print(f"🎯 Target column: {TARGET_COL}")
    print(f"🔢 Using top {TOP_K_FEATURES} features per model")

    print("\n🏗️  Starting training with top-K feature selection...")
    meta_model = train_meta_model(X, y, top_k=TOP_K_FEATURES, n_splits=N_SPLITS)

    print("\n✅ Training complete.")

if __name__ == "__main__":
    main()
