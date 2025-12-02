import cv2
import time
import math
import os
import numpy as np
import joblib
# ---- Protobuf compatibility patch for Mediapipe ----
from google.protobuf import message_factory as _message_factory

if not hasattr(_message_factory, "GetMessageClass"):
    def GetMessageClass(descriptor):
        factory = _message_factory.MessageFactory()
        return factory.GetPrototype(descriptor)
    _message_factory.GetMessageClass = GetMessageClass
# ----------------------------------------------------

import mediapipe as mp

# ==========================
# MEDIAPIPE SETUP
# ==========================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# ==========================
# HAND-SIGN CONFIG
# ==========================
# Only trust hand-signs above this probability
HAND_SIGN_CONFIDENCE_THRESHOLD = 0.90

# Labels that you consider "normal" / harmless
# 👉 change this to match the label name(s) you use in your model
NORMAL_HAND_LABELS = {"normal"}

# Labels that you explicitly consider cheating signals
# 👉 change this to the labels you trained for A,B,C,... / numbers, etc
CHEATING_HAND_LABELS = {"A", "B", "C", "D", "E", "F"}
# e.g. if you add numbers: {"A","B","C","D","E","F","1","2","3","4"}


# ==========================
# LOAD HAND-SIGN CLASSIFIER
# ==========================
try:
    hand_sign_model = joblib.load("hand_sign_mlp.joblib")
    hand_sign_le = joblib.load("hand_sign_label_encoder.joblib")
    print("[INFO] Hand-sign classifier loaded.")
except Exception as e:
    print("[WARN] Could not load hand-sign classifier:", e)
    hand_sign_model = None
    hand_sign_le = None


def extract_hand_features(hand_landmarks):
    """
    Converts MediaPipe 21 hand landmarks to a flat feature vector:
    [x0, y0, z0, x1, y1, z1, ...]
    + simple normalization (subtract wrist).
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                      dtype=np.float32)
    wrist = coords[0].copy()
    coords -= wrist
    return coords.flatten()


# ==========================
# OPTIONAL: YOLO FOR EXTRA PERSON
# ==========================
USE_YOLO = True
yolo_model = None
if USE_YOLO:
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n.pt")  # COCO model (person class = 0)
    except Exception as e:
        print("[WARN] YOLO not available, extra-person detection disabled:", e)
        yolo_model = None

# YOLO filtering thresholds to reduce false positives
EXTRA_PERSON_CONF_THRESH = 0.6      # ignore person boxes below this confidence
EXTRA_PERSON_MIN_AREA_FRAC = 0.05   # ignore tiny boxes (< 5% of frame area)


# ==========================
# SIMPLE BEEP FUNCTION
# ==========================
try:
    import winsound

    def play_beep():
        winsound.Beep(1000, 300)
except Exception:
    def play_beep():
        print("\a")


class CheatingDetector:
    def __init__(self):
        # Heuristic thresholds (tune these!)
        self.visibility_thresh = 0.3
        self.edge_margin = 0.05

        # LEANING / HAND SIGNALS
        self.lean_side_threshold = 0.25
        self.head_low_threshold = 0.7
        self.hand_above_shoulder_margin = 0.03

        # Warning system
        self.warning_delay = 5.0
        self.max_warnings = 3

        # State (pose)
        self.prev_center_x = None

        self.current_violation_start = None
        self.current_violation_types = set()
        self.current_violation_messages = []
        self.warning_count = 0

        # Extra person state
        self.extra_person_current = False
        self.extra_person_ever_seen = False
        self.extra_person_first_time = None
        self.extra_person_duration = 0.0
        self.extra_person_confirm_seconds = 3.0
        self.extra_person_confirmed = False
        self.extra_person_initial_evidence_file = None

        # Statistics
        self.total_frames = 0
        self.violation_frames = 0
        self.violation_history = []

        # Warning events
        self.warning_events = []

        # Evidence folder
        self.evidence_dir = "evidence_frames"
        os.makedirs(self.evidence_dir, exist_ok=True)

    # ---------- Utility ----------
    @staticmethod
    def _dist_1d(a, b):
        return abs(a - b)

    @staticmethod
    def _dist_2d(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _is_landmark_visible(self, l):
        return (
            l.visibility > self.visibility_thresh and
            self.edge_margin <= l.x <= 1.0 - self.edge_margin and
            self.edge_margin <= l.y <= 1.0 - self.edge_margin
        )

    # ---------- Heuristics ----------
    def check_head_and_hands_visible(self, lm):
        nose = lm[0]
        left_wr = lm[15]
        right_wr = lm[16]

        head_visible = self._is_landmark_visible(nose)
        left_visible = self._is_landmark_visible(left_wr)
        right_visible = self._is_landmark_visible(right_wr)

        missing_details = []

        if not head_visible:
            missing_details.append("Head not visible (out of frame or low visibility)")
        if not left_visible:
            missing_details.append("Left hand not visible")
        if not right_visible:
            missing_details.append("Right hand not visible")

        if (not head_visible) or (not left_visible) or (not right_visible):
            msg = ", ".join(missing_details) if missing_details else "Head and/or hands not clearly visible"
            return False, msg

        return True, ""

    def check_leaning(self, lm):
        left_sh = lm[11]
        right_sh = lm[12]
        nose = lm[0]

        center_x = (left_sh.x + right_sh.x) / 2.0
        leaning_reasons = []

        if center_x < self.lean_side_threshold:
            leaning_reasons.append("Leaning to the left (near frame edge)")
        elif center_x > 1 - self.lean_side_threshold:
            leaning_reasons.append("Leaning to the right (near frame edge)")

        if nose.y > self.head_low_threshold:
            leaning_reasons.append("Leaning down (head too low in frame)")

        if self.prev_center_x is not None:
            dx = self._dist_1d(center_x, self.prev_center_x)
            if dx > 0.08:
                leaning_reasons.append("Sudden leaning movement")

        self.prev_center_x = center_x

        if leaning_reasons:
            return False, "; ".join(leaning_reasons)
        return True, ""

    def check_hand_signals(self, lm):
        """
        Pure geometric heuristic: is hand raised high relative to the shoulder?
        This is still used as a general suspicious pattern (e.g., waving).
        """
        left_sh = lm[11]
        right_sh = lm[12]
        left_wr = lm[15]
        right_wr = lm[16]

        reasons = []

        if left_wr.y < left_sh.y - self.hand_above_shoulder_margin:
            reasons.append("Left hand raised (possible signalling)")
        if right_wr.y < right_sh.y - self.hand_above_shoulder_margin:
            reasons.append("Right hand raised (possible signalling)")

        if reasons:
            return False, "; ".join(reasons)
        return True, ""

    def is_any_hand_raised(self, lm):
        """
        Helper for classifier gating.
        Returns True if at least one hand is clearly raised above its shoulder.
        This is used together with the classifier, so it's OK to be a bit strict.
        """
        left_sh = lm[11]
        right_sh = lm[12]
        left_wr = lm[15]
        right_wr = lm[16]

        left_raised = left_wr.y < left_sh.y - self.hand_above_shoulder_margin
        right_raised = right_wr.y < right_sh.y - self.hand_above_shoulder_margin

        return left_raised or right_raised

    def save_evidence_frame(self, frame, messages):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.evidence_dir,
                                f"warning_{self.warning_count}_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved evidence frame: {filename}")
        return filename

    def check_extra_person(self, frame):
        """
        Use YOLO to detect extra persons in the frame.
        Filters out:
        - low-confidence boxes
        - very small boxes (tiny blobs, noise)
        """
        if yolo_model is None:
            self.extra_person_current = False
            self.extra_person_duration = 0.0
            self.extra_person_first_time = None
            return False, ""

        results = yolo_model(frame, verbose=False)
        if not results:
            self.extra_person_current = False
            self.extra_person_duration = 0.0
            self.extra_person_first_time = None
            return False, ""

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            self.extra_person_current = False
            self.extra_person_duration = 0.0
            self.extra_person_first_time = None
            return False, ""

        h, w, _ = frame.shape
        frame_area = float(w * h)

        num_persons = 0
        for b in boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])

            # keep only "person" class
            if cls_id != 0:
                continue

            # filter on confidence
            if conf < EXTRA_PERSON_CONF_THRESH:
                continue

            x1, y1, x2, y2 = b.xyxy[0]
            box_w = float(x2 - x1)
            box_h = float(y2 - y1)
            box_area = box_w * box_h
            area_frac = box_area / frame_area if frame_area > 0 else 0.0

            # ignore very small boxes (likely artifacts)
            if area_frac < EXTRA_PERSON_MIN_AREA_FRAC:
                continue

            num_persons += 1

        now = time.time()

        if num_persons > 1:
            # At least 2 real, big, confident person boxes
            if not self.extra_person_current:
                self.extra_person_current = True
                self.extra_person_first_time = now
                if not self.extra_person_ever_seen:
                    self.extra_person_ever_seen = True
                    self.extra_person_initial_evidence_file = self.save_evidence_frame(
                        frame, ["Initial extra person detection"]
                    )
                    print("[INFO] Extra person first seen, initial evidence saved.")

            if self.extra_person_first_time is None:
                self.extra_person_first_time = now
            self.extra_person_duration = now - self.extra_person_first_time

            if (self.extra_person_duration >= self.extra_person_confirm_seconds
                    and not self.extra_person_confirmed):
                self.extra_person_confirmed = True
                print(f"[INFO] Extra person confirmed (duration >= {self.extra_person_confirm_seconds}s).")

            msg = (
                f"Extra person detected in frame (count={num_persons}, "
                f"current duration≈{self.extra_person_duration:.1f}s)"
            )
            return True, msg
        else:
            self.extra_person_current = False
            self.extra_person_duration = 0.0
            self.extra_person_first_time = None
            return False, ""

    def evaluate_frame(self, frame, pose_results):
        self.total_frames += 1
        violation_types = set()
        violation_messages = []

        # Extra person
        extra_flag, extra_msg = self.check_extra_person(frame)
        if extra_flag:
            violation_types.add("extra_person")
            violation_messages.append(extra_msg)

        # Pose-related checks
        if not pose_results.pose_landmarks:
            violation_types.add("not_visible")
            violation_messages.append("Student not visible (pose not detected)")
        else:
            lm = pose_results.pose_landmarks.landmark

            ok_vis, msg_vis = self.check_head_and_hands_visible(lm)
            if not ok_vis:
                violation_types.add("visibility")
                violation_messages.append(msg_vis)

            ok_lean, msg_lean = self.check_leaning(lm)
            if not ok_lean:
                violation_types.add("leaning")
                violation_messages.append(msg_lean)

            ok_hand, msg_hand = self.check_hand_signals(lm)
            if not ok_hand:
                violation_types.add("hand_signals")
                violation_messages.append(msg_hand)

        if violation_types:
            self.violation_frames += 1
            self.violation_history.append((time.time(), list(violation_types)))

        return violation_types, violation_messages

    def update_warnings(self, violation_types, violation_messages, frame):
        now = time.time()

        if violation_types:
            if not self.current_violation_types:
                self.current_violation_start = now
                self.current_violation_types = violation_types.copy()
                self.current_violation_messages = violation_messages.copy()
                play_beep()
            else:
                self.current_violation_types |= violation_types
                for m in violation_messages:
                    if m not in self.current_violation_messages:
                        self.current_violation_messages.append(m)

                elapsed = now - self.current_violation_start
                if elapsed >= self.warning_delay:
                    self.warning_count += 1
                    evidence_file = self.save_evidence_frame(
                        frame, self.current_violation_messages)
                    self.warning_events.append({
                        "time": now,
                        "types": self.current_violation_types.copy(),
                        "messages": self.current_violation_messages.copy(),
                        "file": evidence_file
                    })
                    print(f"[WARN] Warning #{self.warning_count} registered.")
                    play_beep()
                    self.current_violation_start = now
        else:
            self.current_violation_types = set()
            self.current_violation_messages = []
            self.current_violation_start = None

    def compute_final_probability(self):
        if self.extra_person_confirmed:
            expl = f"Extra person stayed in frame for at least {self.extra_person_confirm_seconds} seconds."
            return 1.0, expl

        if self.total_frames == 0:
            return 0.0, "No frames processed."

        violation_ratio = self.violation_frames / self.total_frames
        warning_ratio = self.warning_count / max(1, self.max_warnings)

        prob = 0.7 * warning_ratio + 0.3 * violation_ratio
        prob = min(prob, 0.85)
        prob = max(0.0, min(1.0, prob))

        explanation = (
            f"Warnings: {self.warning_count}/{self.max_warnings}, "
            f"Violation frames: {self.violation_frames}/{self.total_frames} "
            f"({violation_ratio:.2f})."
        )

        if self.extra_person_ever_seen and not self.extra_person_confirmed:
            explanation += (
                f" Extra person was briefly seen in the frame (duration < {self.extra_person_confirm_seconds}s); "
                "please review evidence."
            )

        return prob, explanation

    def describe_warning_events(self):
        if not self.warning_events:
            return ["No warnings were issued during this session."]

        descriptions = []
        for i, ev in enumerate(self.warning_events, start=1):
            msgs = ev.get("messages", [])
            file = ev.get("file", "N/A")
            joined = "; ".join(msgs) if msgs else "Unspecified violations"
            descriptions.append(f"Warning #{i}: {joined} | Evidence: {file}")
        return descriptions


def main():
    cap = cv2.VideoCapture(0)
    detector = CheatingDetector()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose, mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_pose = pose.process(rgb)
            results_hands = hands.process(rgb)

            # -------- Hand sign classification (with gating) --------
            recognized_sign = None
            recognized_conf = None
            hand_raised_for_signal = False

            # Check if any hand is raised according to pose
            if results_pose.pose_landmarks:
                lm_pose = results_pose.pose_landmarks.landmark
                hand_raised_for_signal = detector.is_any_hand_raised(lm_pose)

            if (
                hand_sign_model is not None and
                results_hands.multi_hand_landmarks and
                hand_raised_for_signal  # only classify if hand is raised
            ):
                hand_lm = results_hands.multi_hand_landmarks[0]
                feats = extract_hand_features(hand_lm).reshape(1, -1)

                probs = hand_sign_model.predict_proba(feats)[0]
                class_idx = np.argmax(probs)
                conf = probs[class_idx]
                label = hand_sign_le.inverse_transform([class_idx])[0]

                # Only treat as a cheating sign if:
                # 1) high confidence,
                # 2) label is not a "normal" label, and
                # 3) label is one of the explicit cheating labels (e.g. A–F)
                if (
                    conf >= HAND_SIGN_CONFIDENCE_THRESHOLD and
                    label not in NORMAL_HAND_LABELS and
                    label in CHEATING_HAND_LABELS
                ):
                    recognized_sign = label
                    recognized_conf = conf

            # -------- Pose + YOLO heuristics --------
            violation_types, violation_messages = detector.evaluate_frame(
                frame, results_pose
            )

            # Only count explicit hand sign as violation if it passed gating
            if recognized_sign is not None and hand_raised_for_signal:
                violation_types.add("explicit_hand_sign")
                violation_messages.append(
                    f"Recognized hand sign '{recognized_sign}' (conf {recognized_conf:.2f})"
                )

            detector.update_warnings(violation_types, violation_messages, frame)

            # -------- Overlay UI --------
            y0 = 30
            dy = 25

            status_text = f"Warnings: {detector.warning_count}/{detector.max_warnings}"
            if detector.extra_person_confirmed:
                status_text += " | EXTRA PERSON CONFIRMED"
            elif detector.extra_person_current:
                status_text += " | EXTRA PERSON PRESENT"
            cv2.putText(frame, status_text, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255 if detector.warning_count > 0 else 255), 2)

            if 'extra_person' in violation_types:
                cv2.putText(frame, "🚨 EXTRA PERSON DETECTED 🚨",
                            (10, y0 + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
                cv2.putText(frame, "Another person is visible. They must leave the frame.",
                            (10, y0 + 2*dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)

            # Show sign only if it's considered a cheating signal & hand raised
            if recognized_sign is not None and hand_raised_for_signal:
                cv2.putText(frame, f"Sign: {recognized_sign} ({recognized_conf:.2f})",
                            (10, y0 + 3*dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 0, 0), 2)

            if violation_types:
                y = y0 + 4*dy if 'extra_person' in violation_types else y0 + 2*dy
                cv2.putText(frame, "⚠ Suspicious behavior detected!",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y += dy
                cv2.putText(frame, "Please adjust your position:",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y += dy

                for msg in violation_messages:
                    short_msg = msg[:60]
                    cv2.putText(frame, f"- {short_msg}", (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    y += dy

                if detector.current_violation_start is not None:
                    remaining = detector.warning_delay - (time.time() - detector.current_violation_start)
                    if remaining < 0:
                        remaining = 0
                    y += dy
                    cv2.putText(frame, f"Warning in: {remaining:.1f} s",
                                (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.imshow("Pose-based Cheating Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    print("\n========== SESSION SUMMARY ==========")
    warning_descriptions = detector.describe_warning_events()
    print("Warnings detail:")
    for line in warning_descriptions:
        print("  -", line)

    prob, explanation = detector.compute_final_probability()

    if prob < 0.3:
        status = "Clean"
    elif prob < 0.6:
        status = "Suspicious"
    else:
        status = "Highly suspicious"

    print(f"\nCheating probability: {prob * 100:.1f}% ({status})")
    print(f"Explanation: {explanation}")
    if detector.extra_person_confirmed:
        print("Reason: Extra person remained in frame beyond "
              f"{detector.extra_person_confirm_seconds} seconds.")


if __name__ == "__main__":
    main()