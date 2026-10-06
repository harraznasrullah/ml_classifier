"""
Real-Time Sign Language Recognition - Enhanced Version
Shows prediction with confidence bar and smoothing

Uses MediaPipe Tasks API (newer version compatible with Python 3.13)
"""
import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter

# Load the trained model and labels
model = joblib.load("model.pkl")
label_encoder = joblib.load("labels.pkl")

# MediaPipe Hands (using Tasks API)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def create_hand_landmarker():
    """Create and configure the hand landmarker."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    return HandLandmarker.create_from_options(options)


# Create the landmarker
landmarker = create_hand_landmarker()

# Camera setup
cap = cv2.VideoCapture(0)

# Buffer for smoothing predictions (keeps last 15 predictions)
prediction_buffer = deque(maxlen=15)
frame_timestamp_ms = 0

print("Running... Press Q to quit")

while True:
    # Get frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)
    frame_timestamp_ms += 1

    # Convert to RGB for MediaPipe (using Image class)
    rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_data)

    label_text = "No hand"
    confidence_value = 0.0

    # Detect hands
    result = landmarker.detect_for_video(rgb_frame, frame_timestamp_ms)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Draw landmarks manually
            for lm in hand_landmarks:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw connections between landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # index
                (0, 9), (9, 10), (10, 11), (11, 12),  # middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
                (5, 9), (9, 13), (13, 17)  # palm
            ]
            for start, end in connections:
                start_lm = hand_landmarks[start]
                end_lm = hand_landmarks[end]
                x1, y1 = int(start_lm.x * frame.shape[1]), int(start_lm.y * frame.shape[0])
                x2, y2 = int(end_lm.x * frame.shape[1]), int(end_lm.y * frame.shape[0])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Extract landmarks (x, y, z for each point)
            row = []
            for lm in hand_landmarks:
                row.extend([lm.x, lm.y, lm.z])

            X = np.array(row).reshape(1, -1)

            # Get prediction and confidence
            probs = model.predict_proba(X)[0]
            pred_index = np.argmax(probs)
            confidence_value = probs[pred_index]

            # Convert to label name
            label = label_encoder.inverse_transform([pred_index])[0]

            # Add to buffer
            prediction_buffer.append(label)

            # Get most common prediction from buffer (smoothing)
            if prediction_buffer:
                most_common = Counter(prediction_buffer).most_common(1)[0][0]

                # Only show if confidence is high enough
                if confidence_value > 0.75:
                    label_text = most_common
                else:
                    label_text = "..."

    # ---------------------------
    # UI: PREDICTION TEXT
    # ---------------------------
    cv2.putText(
        frame,
        f"Prediction: {label_text}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    # ---------------------------
    # UI: CONFIDENCE BAR
    # ---------------------------

    bar_x = 10
    bar_y = 90
    bar_width = 300
    bar_height = 20

    # Background bar (gray)
    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1
    )

    # Filled bar (green) based on confidence
    fill_width = int(bar_width * confidence_value)

    cv2.rectangle(
        frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 255, 0), -1
    )

    # Confidence percentage text
    cv2.putText(
        frame,
        f"Confidence: {confidence_value * 100:.0f}%",
        (bar_x, bar_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )

    # Display the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Press Q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
