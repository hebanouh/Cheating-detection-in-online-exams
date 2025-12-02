from ultralytics import YOLO
import cv2, os, csv, time
import numpy as np
from datetime import datetime
import mediapipe as mp


def run_detect_object(model_path="NewFullDataSet_FT.pt"):
    """Runs the monitoring system once. Press Q to exit."""

    # ================= YOLO MODEL =================
    model = YOLO(model_path)

    # ================= MediaPipe POSE =================
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_landmarks(frame, pose_result):
        """Returns wrist and ear positions (pixel coordinates)."""
        h, w, _ = frame.shape
        if not pose_result.pose_landmarks:
            return [], []

        lm = pose_result.pose_landmarks.landmark

        # Wrist points
        wrists = []
        if lm[mp_pose.PoseLandmark.LEFT_WRIST].visibility > 0.50:
            wrists.append((int(lm[mp_pose.PoseLandmark.LEFT_WRIST].x * w),
                           int(lm[mp_pose.PoseLandmark.LEFT_WRIST].y * h)))
        if lm[mp_pose.PoseLandmark.RIGHT_WRIST].visibility > 0.50:
            wrists.append((int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].x * w),
                           int(lm[mp_pose.PoseLandmark.RIGHT_WRIST].y * h)))

        # Ear points
        ears = []
        if lm[mp_pose.PoseLandmark.LEFT_EAR].visibility > 0.50:
            ears.append((int(lm[mp_pose.PoseLandmark.LEFT_EAR].x * w),
                         int(lm[mp_pose.PoseLandmark.LEFT_EAR].y * h)))
        if lm[mp_pose.PoseLandmark.RIGHT_EAR].visibility > 0.50:
            ears.append((int(lm[mp_pose.PoseLandmark.RIGHT_EAR].x * w),
                         int(lm[mp_pose.PoseLandmark.RIGHT_EAR].y * h)))
        return wrists, ears

    def refine_label(yolo_label, conf, center, wrists, ears):
        """Refines low confidence predictions using proximity logic."""
        print("⚠ YOLO ORIGINAL:", yolo_label, "| Conf:", conf)  # <--- debug
        cx, cy = center
        if yolo_label not in ["headphones", "watch"]:
            return yolo_label
        if conf >= 0.60:
            return yolo_label
        # Near wrist -> watch
        if wrists:
            print("wrists")
            distances = [np.linalg.norm(np.array((cx, cy)) - np.array(w)) for w in wrists]
            if min(distances) < 90:
                return "watch"
        # Near ears -> headphones
        if ears:
            print("ears")
            distances = [np.linalg.norm(np.array((cx, cy)) - np.array(e)) for e in ears]
            if min(distances) < 120:
                return "headphones"
        print("hiiiiiii",yolo_label)
        return yolo_label

    # ================= CONFIG =================
    FORBIDDEN = ["mobile", "book", "watch", "headphones", "sunglass", "earbuds", "laptop", "face_mask"]

    SUNGLASS_THRESHOLD = 0.86
    EARBUDS_THRESHOLD = 0.47
    HEADPHONES_THRESHOLD = 0.50
    WATCH_THRESHOLD = 0.30
    DEFAULT_THRESHOLD = 0.50
    DELAY_SECONDS = 0.75

    # ================= FOLDER SETUP =================
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    evidence_folder = f"evidence_{run_id}"
    os.makedirs(evidence_folder, exist_ok=True)

    report_path = f"{evidence_folder}/exam_report_{run_id}.csv"
    with open(report_path, "w", newline="") as f:
        csv.writer(f).writerow(["Time", "Violation", "Confidence", "Image"])

    detected_once = set()
    detect_timer = {}

    cap = cv2.VideoCapture(0)

    print(" Monitoring Started... Press Q to exit.")


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose_result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        wrists, ears = get_landmarks(frame, pose_result)

        results = model(frame)
        current_detected = set()

        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower().strip()
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            refined_label = refine_label(label, conf, (cx, cy), wrists, ears)
            print(f"Refined: {label} → {refined_label} | Conf: {conf:.2f}")

            if refined_label not in FORBIDDEN:
                continue

            current_detected.add(refined_label)

            if refined_label in detected_once and refined_label != "watch":
                continue

            base_threshold = {
                "sunglass": SUNGLASS_THRESHOLD,
                "earbuds": EARBUDS_THRESHOLD,
                "headphones": HEADPHONES_THRESHOLD,
                "watch": WATCH_THRESHOLD
            }.get(refined_label, DEFAULT_THRESHOLD)

            box_area = (x2 - x1) * (y2 - y1)
            frame_area = frame.shape[0] * frame.shape[1]
            relative_area = box_area / frame_area
            adjusted_threshold = base_threshold * (0.5 + 0.5 * relative_area / 0.02)
            adjusted_threshold = min(max(adjusted_threshold, 0.3), 0.9)

            # ================= INSTANT DETECT =================
            if refined_label in ["watch", "mobile", "headphones"] and (
                    conf >= adjusted_threshold or refined_label != label):
                filename = f"{evidence_folder}/{refined_label}_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)

                with open(report_path, "a", newline="") as f:
                    csv.writer(f).writerow([datetime.now(), refined_label, conf, filename])

                print(f" [INSTANT] {refined_label} recorded!")
                detected_once.add(refined_label)
                continue

            # ================= DELAY DETECT =================
            if conf >= adjusted_threshold or (label != refined_label):
                if refined_label not in detect_timer:
                    detect_timer[refined_label] = time.time()
                else:
                    if time.time() - detect_timer[refined_label] >= DELAY_SECONDS:
                        filename = f"{evidence_folder}/{refined_label}_{int(time.time())}.jpg"
                        cv2.imwrite(filename, frame)

                        with open(report_path, "a", newline="") as f:
                            csv.writer(f).writerow([datetime.now(), refined_label, conf, filename])

                        print(f" [ALERT] {refined_label} logged!")
                        detected_once.add(refined_label)
            else:
                detect_timer.pop(refined_label, None)

        # Reset timers if object disappears
        for obj in list(detect_timer.keys()):
            if obj not in current_detected:
                detect_timer.pop(obj, None)

        cv2.imshow("Exam Monitoring System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n Monitoring stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n Evidence saved in: {evidence_folder}")
    print(f" Report: {report_path}")

# ----------- RUN FROM MAIN -----------
if __name__ == "__main__":
    run_detect_object()