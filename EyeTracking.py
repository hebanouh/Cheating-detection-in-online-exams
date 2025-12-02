import cv2
import mediapipe as mp
import numpy as np
import os, csv, time
from datetime import datetime


def run_eye_tracking():

    # ===========================
    # Folder Setup
    # ===========================
    gaze_folder = "gaze_screenshots"
    blink_folder = "blink_screenshots"
    report_path = "eye_tracking_report.csv"

    if not os.path.exists(gaze_folder):
        os.makedirs(gaze_folder)

    if not os.path.exists(blink_folder):
        os.makedirs(blink_folder)

    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    LEFT_IRIS = [468, 469, 470, 471]
    RIGHT_IRIS = [473, 474, 475, 476]

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

    # Create CSV Header if not exists
    if not os.path.exists(report_path):
        with open(report_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Gaze", "EAR", "Blink", "Screenshot", "Suspicious"])

    # ===========================
    # Helper Functions
    # ===========================
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def get_iris_center(landmarks, indices, w, h):
        pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
        return np.mean(pts, axis=0)

    def gaze_ratio(iris_center, corner_left, corner_right):
        eye_width = np.linalg.norm(corner_right - corner_left)
        iris_x = np.linalg.norm(iris_center - corner_left)
        return iris_x / eye_width

    # ===========================
    # GAZE TIMING VARIABLES
    # ===========================
    last_gaze = "Center"
    gaze_start_time = time.time()
    SUSPICIOUS_THRESHOLD = 2  # seconds

    # ===========================
    # CAMERA LOOP
    # ===========================
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        gaze = "Unknown"
        blink = False
        ear = 0
        screenshot_file = ""
        suspicious_event = "No"

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # EAR = blink detection
            left_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in LEFT_EYE])
            right_eye = np.array([(lm[i].x * w, lm[i].y * h) for i in RIGHT_EYE])

            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

            # Blink detection
            if ear < 0.18:
                blink = True
                screenshot_file = f"{blink_folder}/blink_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_file, frame)

            # Gaze estimation
            left_iris_center = get_iris_center(lm, LEFT_IRIS, w, h)
            right_iris_center = get_iris_center(lm, RIGHT_IRIS, w, h)

            L_left = np.array([lm[33].x * w, lm[33].y * h])
            L_right = np.array([lm[133].x * w, lm[133].y * h])

            R_left = np.array([lm[362].x * w, lm[362].y * h])
            R_right = np.array([lm[263].x * w, lm[263].y * h])

            left_ratio = gaze_ratio(left_iris_center, L_left, L_right)
            right_ratio = gaze_ratio(right_iris_center, R_left, R_right)
            ratio = (left_ratio + right_ratio) / 2

            if ratio < 0.40:
                gaze = "Right"
            elif ratio > 0.60:
                gaze = "Left"
            else:
                gaze = "Center"

            # Gaze timing logic
            if gaze != last_gaze:
                last_gaze = gaze
                gaze_start_time = time.time()

            gaze_duration = time.time() - gaze_start_time

            # Suspicious gaze
            if gaze in ["Left", "Right"] and gaze_duration >= SUSPICIOUS_THRESHOLD:
                suspicious_event = "Yes"
                screenshot_file = f"{gaze_folder}/suspicious_{gaze}_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_file, frame)

            elif gaze in ["Left", "Right"]:
                screenshot_file = f"{gaze_folder}/normal_{gaze}_{int(time.time())}.jpg"
                cv2.imwrite(screenshot_file, frame)

            # Log to CSV
            with open(report_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    gaze,
                    f"{ear:.2f}",
                    "Yes" if blink else "No",
                    screenshot_file,
                    suspicious_event
                ])

        cv2.imshow("Exam Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    return report_path, gaze_folder, blink_folder
