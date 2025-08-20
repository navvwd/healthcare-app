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
DATA_PATH = BASE / "anemia.csv"
MODELS_DIR = BASE.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "anemia_model.pkl"

if not DATA_PATH.exists():
    print(f"❌ CSV not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]        # clean header spaces
df = df.dropna()

print("📄 Columns:", list(df.columns))

# Try to find target column automatically
possible_targets = ["Anemia", "anemia", "Result", "result", "Target", "target",
                    "Outcome", "outcome", "label", "Label", "condition", "Condition"]
target_col = next((c for c in possible_targets if c in df.columns), None)
if target_col is None:
    raise ValueError(f"Could not find a target column in {list(df.columns)}. "
                     f"Rename your label column to 'Anemia' or one of: {possible_targets}")

# Convert y to 0/1 if it looks categorical
y_raw = df[target_col]
if y_raw.dtype == "O":
    y = (y_raw.astype(str).str.lower().isin(["anemia", "yes", "true", "1", "positive"])).astype(int)
else:
    # many anemia datasets already have 0/1 or 0..n; map >0 to 1
    y = (y_raw.astype(float) > 0).astype(int)

# Preferred features (use what exists)
preferred = ["Age","Hemoglobin","MCV","MCH","MCHC"]
features = [c for c in preferred if c in df.columns and c != target_col]

# Fallback: all numeric columns except target
if len(features) < 2:
    features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]

if len(features) < 2:
    raise ValueError("Not enough numeric features to train. Need at least 2.")

X = df[features].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test))
print(f"✅ Anemia model accuracy: {acc:.3f} using features: {features}")

# Save a bundle with the model + feature order
joblib.dump({"model": clf, "features": features}, MODEL_PATH)
print(f"💾 Saved: {MODEL_PATH}")
