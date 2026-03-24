"""
Train a Random Forest classifier on FlipPhone trick data.

Usage:
    python fetch_data.py --url ... --key ...   # first, fetch the data
    python train.py                             # then, train
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Trick selection ─────────────────────────────────────────────────
# Only these tricks (classes) will be used for training.
# Comment out or remove entries to exclude them.
SELECTED_TRICKS = [
    "Kickflip",
    "Treflip",
    "FS Shuvit",
]


# ── Feature extraction ──────────────────────────────────────────────


def extract_features(group: pd.DataFrame) -> dict:
    """Extract features from a single recording's sample rows."""
    dt = group["t"].diff().fillna(0) / 1000.0  # seconds

    features = {}

    # Per-axis statistics
    for col in ["ax", "ay", "az", "gx", "gy", "gz"]:
        v = group[col]
        features[f"{col}_mean"] = v.mean()
        features[f"{col}_std"] = v.std()
        features[f"{col}_min"] = v.min()
        features[f"{col}_max"] = v.max()
        features[f"{col}_range"] = v.max() - v.min()

    # Acceleration magnitude
    acc_mag = np.sqrt(group["ax"] ** 2 + group["ay"] ** 2 + group["az"] ** 2)
    features["acc_mag_mean"] = acc_mag.mean()
    features["acc_mag_std"] = acc_mag.std()
    features["acc_mag_max"] = acc_mag.max()

    # Rotation magnitude
    gyro_mag = np.sqrt(group["gx"] ** 2 + group["gy"] ** 2 + group["gz"] ** 2)
    features["gyro_mag_mean"] = gyro_mag.mean()
    features["gyro_mag_std"] = gyro_mag.std()
    features["gyro_mag_max"] = gyro_mag.max()

    # Integrated rotation per axis (total angle in radians)
    for axis in ["gx", "gy", "gz"]:
        features[f"{axis}_total_angle"] = np.trapezoid(group[axis], group["t"] / 1000.0)

    # Duration
    features["duration_s"] = (group["t"].iloc[-1] - group["t"].iloc[0]) / 1000.0

    # Sample count
    features["sample_count"] = len(group)

    return features


# ── Main ─────────────────────────────────────────────────────────────


def main():
    if not os.path.exists(DATA_PATH):
        print("No dataset found. Run fetch_data.py first.")
        return

    print("Loading data …")
    df = pd.read_csv(DATA_PATH)

    # Filter to selected tricks
    df = df[df["trick"].isin(SELECTED_TRICKS)]
    if df.empty:
        print(f"No data found for selected tricks: {SELECTED_TRICKS}")
        return
    print(f"  {len(df)} sample rows, {df['id'].nunique()} recordings, {df['trick'].nunique()} tricks")
    print(f"  Selected tricks: {SELECTED_TRICKS}")

    # Show class distribution
    trick_counts = df.groupby("trick")["id"].nunique()
    print("\nRecordings per trick:")
    for trick, count in trick_counts.items():
        print(f"  {trick}: {count}")

    # Extract features per recording
    print("\nExtracting features …")
    records = []
    for rec_id, group in df.groupby("id"):
        feats = extract_features(group)
        feats["id"] = rec_id
        feats["trick"] = group["trick"].iloc[0]
        records.append(feats)

    feat_df = pd.DataFrame(records)
    feature_cols = [c for c in feat_df.columns if c not in ("id", "trick")]

    X = feat_df[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(feat_df["trick"])

    print(f"  {X.shape[0]} recordings, {X.shape[1]} features, {len(le.classes_)} classes")

    # 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

    print("\nTraining Random Forest …")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n── Classification Report (Test Set) ──")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("── Confusion Matrix (Test Set) ──")
    cm = confusion_matrix(y_test, y_pred)
    # Pretty print
    max_name = max(len(n) for n in le.classes_)
    header = " " * (max_name + 2) + "  ".join(f"{n[:6]:>6}" for n in le.classes_)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"{le.classes_[i]:>{max_name}}  {row_str}")

    # Feature importance
    importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
    print("\n── Top 10 Features ──")
    for name, imp in importances[:10]:
        print(f"  {name:<25} {imp:.4f}")

    # Retrain on all data for the exported model
    clf.fit(X, y)

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"clf": clf, "label_encoder": le, "feature_cols": feature_cols}, f)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
