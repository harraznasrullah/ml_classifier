# Sign Language Recognition

Real-time sign language gesture recognition using hand landmark detection and a neural network classifier.

**👉 New here? Start with [SETUP.md](SETUP.md) for step-by-step instructions!**

---

## Quick Overview

This project uses your webcam to recognize hand gestures in real-time:

1. **Hand Detection** — MediaPipe extracts 21 hand landmarks (x, y, z coordinates) from webcam frames
2. **Classification** — A neural network (MLP) trained on labeled data predicts which gesture you're showing
3. **Real-time Display** — Shows the prediction with a confidence bar and smoothed output

---

## Quick Start

### Option 1: Just Run It (Model Already Included!)

```bash
pip install -r requirements.txt
python updated_realtime_test.py
```

Press **Q** to quit.

---

## Project Files

| File | Purpose |
|------|---------|
| `SETUP.md` | **Beginner-friendly setup guide** |
| `updated_realtime_test.py` | Main app - real-time prediction with confidence bar |
| `realtime_test.py` | Simple prediction version |
| `collect_dataset.py` | Record hand gesture data for training |
| `train_model.py` | Train the ML model on collected data |
| `evaluate_model.py` | Model evaluation - accuracy, confusion matrix |
| `model.pkl` | Trained model (included!) |
| `labels.pkl` | Gesture label names |
| `dataset/data.csv` | Training data (63 features + label per row) |

---

## How It Works

### The Pipeline
```
Webcam Frame → MediaPipe → 21 Landmarks (x,y,z) → MLP Model → Gesture Label
```

### Model Details
- **Type**: MLPClassifier (Multi-Layer Perceptron)
- **Architecture**: 2 hidden layers (128 → 64 neurons)
- **Input**: 63 features (21 landmarks × 3 coordinates)
- **Training**: 80/20 train-test split

### Evaluation Results

| Metric | Score |
|--------|-------|
| Test accuracy | 100% |
| 5-fold cross-validation | 99.77% (±0.91%) |
| Classes | hello, sorry, thankyou, yes |
| Dataset size | 441 samples |

---

## Usage Examples

### Test with webcam
```bash
python updated_realtime_test.py
```

### Collect new gesture data
Edit `label` in `collect_dataset.py`, then:
```bash
python collect_dataset.py
```
Press **S** to save a sample, **Q** to quit.

### Train a new model
```bash
python train_model.py
```

### Evaluate model performance
```bash
python evaluate_model.py
```

---

## Dependencies

Install all at once with:
```bash
pip install -r requirements.txt
```

Individual packages:
- opencv-python - Webcam and image processing
- mediapipe - Hand landmark detection
- scikit-learn - Machine learning model
- pandas - Data handling
- numpy - Numerical operations
- joblib - Model saving/loading

---

## Troubleshooting

**Camera not working?**
- Check if another app is using your webcam
- Try a different USB port

**Import errors?**
- Make sure you installed requirements: `pip install -r requirements.txt`
- Check you're using the correct Python environment

**Model not found?**
- Make sure `model.pkl` and `labels.pkl` are in the project folder

---

## Project Structure

```
ml_classifier/
├── dataset/
│   └── data.csv           # Training data
├── models/                 # (empty, models saved to root)
├── .venv/                  # Virtual environment (created by you)
├── .gitignore              # Files to ignore in git
├── requirements.txt        # Python dependencies
├── SETUP.md                # Beginner setup guide
├── README.md               # This file
├── updated_realtime_test.py  # Main app
├── realtime_test.py        # Simple version
├── collect_dataset.py      # Data collection
├── train_model.py          # Model training
├── evaluate_model.py       # Model evaluation
├── model.pkl               # Trained model
└── labels.pkl              # Label encoder
```
