import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

data = pd.read_csv("dataset/data.csv", header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=500,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("=" * 50)
print("MODEL EVALUATION")
print("=" * 50)

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {len(le.classes_)} classes")
print(f"Classes: {list(le.classes_)}")

acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f} ({acc*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"{'':>12}", end="")
for name in le.classes_:
    print(f"{name:>8}", end="")
print()
for i, name in enumerate(le.classes_):
    print(f"{name:>12}", end="")
    for j in range(len(le.classes_)):
        print(f"{cm[i][j]:>8}", end="")
    print()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y_encoded, cv=skf, scoring="accuracy")
print(f"\n5-fold cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

joblib.dump(model, "model.pkl")
joblib.dump(le, "labels.pkl")
print("\nModel and labels saved.")
