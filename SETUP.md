# Quick Start Guide - For Beginners!

## What You Need Before Starting
- A working webcam
- Python installed on your computer (version 3.9 or newer)
- This project folder opened in VS Code

---

## Step 1: Install Python Extension in VS Code
1. Open VS Code
2. Click the Extensions icon on the left (or press `Ctrl+Shift+X`)
3. Search for "Python"
4. Install the one by Microsoft (the first one)

---

## Step 2: Open This Project in VS Code
1. In VS Code, go to **File** → **Open Folder**
2. Select this folder: `ml_classifier`
3. Click "Select Folder"

---

## Step 3: Set Up Virtual Environment
First, check if Python is installed:
1. Press `Ctrl+Shift+P` (opens command palette)
2. Type "Python: Create Environment"
3. Select "Venv"
4. Select the recommended Python version

**Or do it manually:**
1. Open the terminal in VS Code (press `Ctrl+```)
2. Run this command:
   ```
   python -m venv .venv
   ```
3. Activate it:
   - **Windows**: `.\.venv\Scripts\activate`
   - **Mac/Linux**: `source .venv/bin/activate`

You should see `(.venv)` appear at the start of your terminal line.

---

## Step 4: Install Required Packages
With the virtual environment activated, run:
```
pip install -r requirements.txt
```

This will install:
- opencv-python - for webcam access
- mediapipe - for hand detection
- scikit-learn - for machine learning
- pandas, numpy, joblib - for data handling

---

## Step 5: Run the Application!

Your trained model is already included (`model.pkl`), so you can just run:

```
python updated_realtime_test.py
```

### How to Use:
1. A window will open showing your webcam
2. Show your hand to the camera
3. The app will predict which gesture you're making
4. A confidence bar shows how sure the model is
5. Press **Q** to quit

---

## What Each File Does

| File | What It Does |
|------|--------------|
| `updated_realtime_test.py` | **START HERE!** Main app with webcam & predictions + confidence bar |
| `realtime_test.py` | Simple version without confidence bar |
| `collect_dataset.py` | Record your own hand gestures for training |
| `train_model.py` | Train a new model from collected data |
| `evaluate_model.py` | Test how accurate the model is |
| `model.pkl` | Your trained model (already included!) |
| `labels.pkl` | Label names for predictions |
| `hand_landmarker.task` | MediaPipe hand detection model (included!) |
| `dataset/` | Contains your training data (kept!) |

---

## Troubleshooting

### "No module named 'cv2'" or similar errors
→ Make sure you ran `pip install -r requirements.txt`

### "Camera not opening"
→ Check if another app is using your webcam, or try a different USB port

### "Cannot open model.pkl"
→ Make sure you're in the correct folder when running the command

### "Cannot open hand_landmarker.task"
→ This file should already be included. If missing, download from:
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

### Virtual environment won't activate on Windows
→ Run this in PowerShell first:
   ```
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

---

## Want to Add Your Own Gestures?

1. Open `collect_dataset.py`
2. Edit the `label` variable to your gesture name (e.g., "peace")
3. Run `python collect_dataset.py`
4. Press **S** to save samples, **Q** to quit
5. Repeat for each gesture
6. Run `python train_model.py` to train
7. Run `python updated_realtime_test.py` to test!

---

## Need Help?

Check the main `README.md` for more detailed documentation.

Happy coding! 🚀
