"""
Train the Gesture Recognition Model

This script trains a neural network on your collected hand gesture data.
After training, it saves model.pkl and labels.pkl for use by the real-time app.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("=" * 50)
print("Training Gesture Recognition Model")
print("=" * 50)

# Check if dataset exists
dataset_path = "dataset/data.csv"
if not os.path.exists(dataset_path):
    print("Error: Dataset not found!")
    print(f"Please run collect_dataset.py first to create {dataset_path}")
    exit(1)

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(dataset_path, header=None)
print(f"Loaded {len(data)} samples")

# Split into features (X) and labels (y)
# All columns except the last are features (63 landmarks: 21 points × 3 coords)
# The last column is the label (gesture name)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

print(f"Features: {X.shape[1]}")
print(f"Classes: {len(set(y))} - {list(set(y))}")

# Convert text labels to numbers (hello → 0, yes → 1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create the neural network model
print("\nTraining model...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Two hidden layers with 128 and 64 neurons
    max_iter=500,                  # Maximum training iterations
    random_state=42                # For reproducible results
)

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {acc * 100:.2f}%")

# Save the trained model and label encoder
joblib.dump(model, "model.pkl")
joblib.dump(le, "labels.pkl")

print("\nModel saved successfully!")
print("Files created:")
print("  - model.pkl (trained model)")
print("  - labels.pkl (label encoder)")
print("\nYou can now run: python updated_realtime_test.py")
print("=" * 50)
