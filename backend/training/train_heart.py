from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import sys

BASE = Path(__file__).parent
DATA_PATH = BASE / "heart_cleveland_upload.csv"
MODELS_DIR = BASE.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "heart_model.pkl"

if not DATA_PATH.exists():
    print(f"❌ CSV not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
df = df.replace("?", np.nan).dropna()

print("📄 Columns:", list(df.columns))

# find target
possible_targets = ["condition", "Condition", "target", "Target", "label", "Label"]
target_col = next((c for c in possible_targets if c in df.columns), None)
if target_col is None:
    raise ValueError(f"Could not find a target column in {list(df.columns)}. "
                     f"Rename label to 'condition' or one of: {possible_targets}")

# normalize y to 0/1 (Cleveland: >0 means disease)
y_raw = pd.to_numeric(df[target_col], errors="coerce")
y = (y_raw > 0).astype(int)

# Choose all numeric features except target
features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
if len(features) < 2:
    raise ValueError("Not enough numeric features to train.")

X = df[features].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"✅ Heart model accuracy: {acc:.3f} using features: {features}")

joblib.dump({"model": clf, "features": features}, MODEL_PATH)
print(f"💾 Saved: {MODEL_PATH}")
