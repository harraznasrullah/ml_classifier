"""
Dataset Collector - Record Hand Gesture Data

This script helps you collect training data for new gestures.
Edit the 'label' variable below, then run this script.
Press S to save a hand pose sample, Q to quit.
"""
import cv2
import mediapipe as mp
import csv
import os
import time

# ============================================
# SETTINGS - CHANGE THE LABEL BELOW
# ============================================
# Edit this to the gesture name you want to record
# Examples: "hello", "yes", "no", "thankyou", "peace", etc.
label = "sorry"
# ============================================

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


# Make sure dataset folder exists
os.makedirs("dataset", exist_ok=True)

csv_path = "dataset/data.csv"
file_exists = os.path.isfile(csv_path)

# Open CSV file for appending
csv_file = open(csv_path, mode="a", newline="")
csv_writer = csv.writer(csv_file)

# Create landmarker
landmarker = create_hand_landmarker()

# Open webcam
cap = cv2.VideoCapture(0)

print("=" * 50)
print(f"Recording gesture: {label}")
print("=" * 50)
print("Press S to save a sample (make sure hand is visible)")
print("Press Q to quit")
print("=" * 50)

last_save_time = 0
cooldown = 0.5  # Seconds between saves (prevents accidental double-press)

samples_saved = 0
frame_timestamp_ms = 0

while True:
    # Get frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame (more natural)
    frame = cv2.flip(frame, 1)
    frame_timestamp_ms += 1

    # Convert to RGB for MediaPipe (using Image class)
    rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_data)

    # Process the frame
    result = landmarker.detect_for_video(rgb_frame, frame_timestamp_ms)

    # Draw hand landmarks if detected
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            # Draw landmarks manually (green circles)
            for lm in hand_landmarks:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Draw connections (blue lines)
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

    # Show the frame
    cv2.imshow("Dataset Collector", frame)

    # Handle keyboard input
    key = cv2.waitKey(10) & 0xFF
    current_time = time.time()

    # S - Save sample
    if key == ord("s"):
        if result.hand_landmarks:
            if current_time - last_save_time > cooldown:
                for hand_landmarks in result.hand_landmarks:
                    # Extract all 21 landmarks (x, y, z for each)
                    row = []
                    for lm in hand_landmarks:
                        row.extend([lm.x, lm.y, lm.z])

                    # Add the label at the end
                    row.append(label)
                    csv_writer.writerow(row)

                last_save_time = current_time
                samples_saved += 1
                print(f"Saved sample #{samples_saved} for: {label}")
        else:
            print("No hand detected - can't save!")

    # Q - Quit
    elif key == ord("q"):
        break

# Cleanup
cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("=" * 50)
print(f"Saved {samples_saved} samples for '{label}'")
print("Data saved to:", csv_path)
print("=" * 50)
