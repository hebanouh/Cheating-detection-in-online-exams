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


import mediapipe as mp
#
# # ==========================
# # MEDIAPIPE SETUP
# # ==========================
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# ==========================
# HAND-SIGN CONFIG
# ==========================
# Only trust hand-signs above this probability
HAND_SIGN_CONFIDENCE_THRESHOLD = 0.3
# # ---- Robust gating for far/noisy hands ----
# MIN_HAND_AREA = 0.012      # normalized landmark box area (0..1). tune 0.008–0.02
# MIN_SIGN_CONF = 0.70       # extra confidence gate (in addition to your threshold)
# STABLE_FRAMES = 6          # require same sign for N consecutive frames


NORMAL_HAND_LABELS = {"normal"}

# Labels explicitly considered cheating signals

CHEATING_HAND_LABELS = {"A", "B", "C", "D", "E", "F","0","1","2","3","4","5","6","7","8","9"}



# ==========================
# LOAD HAND-SIGN CLASSIFIER
# ==========================
# def load_hand_sign_models(model_path="hand_sign_mlpf.joblib", le_path="hand_sign_label_encoderf.joblib"):
#     try:
#         model = joblib.load(model_path)
#         le = joblib.load(le_path)
#         print("[INFO] Hand-sign classifier loaded.")
#         return model, le
#     except Exception as e:
#         print("[WARN] Could not load hand-sign classifier:", e)
#         return None, None
def load_hand_sign_models(model_path="hand_sign_mlpf.joblib", le_path="hand_sign_label_encoderf.joblib"):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_path)
        le_path = os.path.join(base_dir, le_path)

        model = joblib.load(model_path)
        le = joblib.load(le_path)
        print("[INFO] Hand-sign classifier loaded.")
        return model, le
    except Exception as e:
        print("[WARN] Could not load hand-sign classifier:", e)
        return None, None

hand_sign_model, hand_sign_le = load_hand_sign_models()



# def extract_hand_features(hand_landmarks):
#     """
#     Converts MediaPipe 21 hand landmarks to a flat feature vector:
#     [x0, y0, z0, x1, y1, z1, ...]
#     + simple normalization (subtract wrist).
#     """
#     coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
#                       dtype=np.float32)
#     wrist = coords[0].copy()
#     coords -= wrist
#     return coords.flatten()
def extract_hand_features(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)

    # 1) translate: wrist at origin
    wrist = coords[0].copy()
    coords -= wrist

    # 2) scale normalize: use wrist->middle_mcp distance (landmark 9)
    scale = np.linalg.norm(coords[9]) + 1e-6
    coords /= scale

    return coords.flatten()


# def hand_area_from_landmarks(hand_lm):
#     xs = [p.x for p in hand_lm.landmark]
#     ys = [p.y for p in hand_lm.landmark]
#     w = max(xs) - min(xs)
#     h = max(ys) - min(ys)
#     return w * h

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
MAX_BEEPS = 3
BEEP_COUNT = 0

try:
    import winsound

    def play_beep():
        global BEEP_COUNT
        if BEEP_COUNT >= MAX_BEEPS:
            return
        winsound.Beep(1000, 300)
        BEEP_COUNT += 1

except Exception:
    def play_beep():
        global BEEP_COUNT
        if BEEP_COUNT >= MAX_BEEPS:
            return
        print("\a")
        BEEP_COUNT += 1


class CheatingDetector:
    def __init__(self, evidence_dir):
        # Store last detected EXTRA boxes (xyxy + conf)
        self.last_extra_person_boxes = []  # list of dicts: {"xyxy": (x1,y1,x2,y2), "conf": float}

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
        self.evidence_dir = evidence_dir

        # # Hand-sign stability state
        # self._prev_sign_label = None
        # self._stable_sign_count = 0
        # -------------------------------
        # Event-based evidence (ONE shot per event)
        # -------------------------------
        self.event_states = {}  # key -> {active, start, evidence_taken, last_seen}
        self.event_log = []     # list of {time, key, duration, file, messages}

        # How long a violation must persist before we take ONE evidence
        # (tune as you like; 0.0 means "instant once")
        self.event_confirm_seconds = {
            "leaning": 2.0,
            "visibility": 2.0,
            "pose_not_detected": 2.0,
            "out_of_frame": 2.0,
            "hand_signals": 1.5,
            "extra_person": 3.0,          # aligns with your existing logic
            "explicit_hand_sign": 0.0     # take once immediately when recognized
        }

        # Optional: prevent re-triggering too quickly after it ends
        self.event_cooldown_seconds = 1.0


    def save_hand_sign_evidence(self, frame, sign_label, conf):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.evidence_dir,
            f"hand_sign_{sign_label}_{conf:.2f}_{timestamp}.jpg"
        )
        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved hand-sign evidence: {filename}")
        return filename

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
    def check_head_and_hands_visible(self, pose_landmarks):
        nose = pose_landmarks[0]
        left_wr = pose_landmarks[15]
        right_wr = pose_landmarks[16]

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

    def check_leaning(self, pose_landmarks):
        left_sh = pose_landmarks[11]
        right_sh = pose_landmarks[12]
        nose = pose_landmarks[0]

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

    def check_hand_signals(self, pose_landmarks):
        """
        Pure geometric heuristic: is hand raised high relative to the shoulder?
        This is still used as a general suspicious pattern (e.g., waving).
        """
        left_sh = pose_landmarks[11]
        right_sh = pose_landmarks[12]
        left_wr = pose_landmarks[15]
        right_wr = pose_landmarks[16]

        reasons = []

        if left_wr.y < left_sh.y - self.hand_above_shoulder_margin:
            reasons.append("Left hand raised (possible signalling)")
        if right_wr.y < right_sh.y - self.hand_above_shoulder_margin:
            reasons.append("Right hand raised (possible signalling)")

        if reasons:
            return False, "; ".join(reasons)
        return True, ""

    def save_evidence_frame(self, frame, messages, prefix="evidence"):

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.evidence_dir, f"{prefix}_{self.warning_count}_{timestamp}.jpg")

        # Optional: write 1–2 messages on the frame for quick review
        try:
            y = 25
            for msg in messages[:3]:
                cv2.putText(frame, msg[:70], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                y += 22
        except:
            pass

        cv2.imwrite(filename, frame)
        print(f"[INFO] Saved evidence frame: {filename}")
        return filename

    # def is_any_hand_raised(self, pose_landmarks):
    #     """
    #     Helper for classifier gating.
    #     Returns True if at least one hand is clearly raised above its shoulder.
    #     This is used together with the classifier, so it's OK to be a bit strict.
    #     """
    #     left_sh = pose_landmarks[11]
    #     right_sh = pose_landmarks[12]
    #     left_wr = pose_landmarks[15]
    #     right_wr = pose_landmarks[16]
    #
    #     left_raised = left_wr.y < left_sh.y - self.hand_above_shoulder_margin
    #     right_raised = right_wr.y < right_sh.y - self.hand_above_shoulder_margin
    #
    #     return left_raised or right_raised
    def classify_hand_sign(self, pose_results, hands_results):
        """
        Returns (label, confidence) or (None, None)
        """
        if hand_sign_model is None or hand_sign_le is None:
            return None, None
        if not pose_results.pose_landmarks:
            return None, None
        if not hands_results.multi_hand_landmarks:
            return None, None

        pose_lm = pose_results.pose_landmarks.landmark
        # if not self.is_any_hand_raised(pose_lm):
        #     return None, None

        hand_lm = hands_results.multi_hand_landmarks[0]
        feats = extract_hand_features(hand_lm).reshape(1, -1)

        probs = hand_sign_model.predict_proba(feats)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = hand_sign_le.inverse_transform([idx])[0]

        if conf >= HAND_SIGN_CONFIDENCE_THRESHOLD and label not in NORMAL_HAND_LABELS and label in CHEATING_HAND_LABELS:
            return label, conf

        return None, None

    def classify_hand_sign_with_roi(self, frame, pose_results, hands,
                                    roi_scale=2.8, min_roi=140, input_size=320):
        """
        ROI-based hand-sign recognition for far hands.
        Builds a crop around each wrist using pose landmarks, resizes it,
        then runs MediaPipe Hands on the crop.

        Returns (label, confidence) or (None, None)
        """
        if hand_sign_model is None or hand_sign_le is None:
            return None, None
        if frame is None or frame.size == 0:
            return None, None
        if not pose_results or not pose_results.pose_landmarks:
            return None, None

        pose_lm = pose_results.pose_landmarks.landmark
        # if not self.is_any_hand_raised(pose_lm):
        #     return None, None

        h, w = frame.shape[:2]

        def _roi_from_wrist_elbow(wrist_idx, elbow_idx):
            wrist = pose_lm[wrist_idx]
            elbow = pose_lm[elbow_idx]

            wx, wy = int(wrist.x * w), int(wrist.y * h)
            ex, ey = int(elbow.x * w), int(elbow.y * h)

            if wx <= 0 or wy <= 0 or wx >= w or wy >= h:
                return None

            dist = math.hypot(wx - ex, wy - ey)
            size = int(max(min_roi, dist * roi_scale))

            # shift ROI a bit past the wrist (towards the hand direction)
            vx, vy = (wx - ex), (wy - ey)
            cx = int(wx + 0.35 * vx)
            cy = int(wy + 0.35 * vy)

            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, cx + size // 2)
            y2 = min(h, cy + size // 2)

            if (x2 - x1) < 40 or (y2 - y1) < 40:
                return None
            return (x1, y1, x2, y2)

        rois = []
        # Left (wrist=15, elbow=13)
        r = _roi_from_wrist_elbow(15, 13)
        if r: rois.append(r)
        # Right (wrist=16, elbow=14)
        r = _roi_from_wrist_elbow(16, 14)
        if r: rois.append(r)

        best_label, best_conf = None, None

        for (x1, y1, x2, y2) in rois:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_rgb = cv2.resize(crop_rgb, (input_size, input_size),
                                  interpolation=cv2.INTER_LINEAR)

            crop_hands = hands.process(crop_rgb)
            if not crop_hands.multi_hand_landmarks:
                continue

            hand_lm = crop_hands.multi_hand_landmarks[0]
            feats = extract_hand_features(hand_lm).reshape(1, -1)

            probs = hand_sign_model.predict_proba(feats)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = hand_sign_le.inverse_transform([idx])[0]

            if conf >= HAND_SIGN_CONFIDENCE_THRESHOLD and label not in NORMAL_HAND_LABELS and label in CHEATING_HAND_LABELS:
                if best_conf is None or conf > best_conf:
                    best_label, best_conf = label, conf

        return best_label, best_conf

    # def save_evidence_frame(self, frame, messages, prefix="evidence"):
    #     timestamp = time.strftime("%Y%m%d_%H%M%S")
    #     filename = os.path.join(self.evidence_dir, f"{prefix}_{self.warning_count}_{timestamp}.jpg")
    #     cv2.imwrite(filename, frame)
    #     print(f"[INFO] Saved evidence frame: {filename}")
    #     return filename

        # def classify_hand_sign_with_roi(self, frame, pose_results, hands,
    #                                 roi_scale=2.8, min_roi=140, input_size=320):
    #     """
    #     ROI-based hand-sign recognition for far hands.
    #     Adds:
    #     (1) hand-size gate
    #     (2) confidence gate
    #     (3) stability gate (N consecutive frames)
    #     Returns (label, confidence) or (None, None)
    #     """
    #     if hand_sign_model is None or hand_sign_le is None:
    #         return None, None
    #     if frame is None or frame.size == 0:
    #         return None, None
    #     if not pose_results or not pose_results.pose_landmarks:
    #         return None, None
    #
    #     pose_lm = pose_results.pose_landmarks.landmark
    #     h, w = frame.shape[:2]

        def _roi_from_wrist_elbow(wrist_idx, elbow_idx):
            wrist = pose_lm[wrist_idx]
            elbow = pose_lm[elbow_idx]

            wx, wy = int(wrist.x * w), int(wrist.y * h)
            ex, ey = int(elbow.x * w), int(elbow.y * h)

            if wx <= 0 or wy <= 0 or wx >= w or wy >= h:
                return NoneS

            dist = math.hypot(wx - ex, wy - ey)
            size = int(max(min_roi, dist * roi_scale))

            vx, vy = (wx - ex), (wy - ey)
            cx = int(wx + 0.35 * vx)
            cy = int(wy + 0.35 * vy)

            x1 = max(0, cx - size // 2)
            y1 = max(0, cy - size // 2)
            x2 = min(w, cx + size // 2)
            y2 = min(h, cy + size // 2)

            if (x2 - x1) < 40 or (y2 - y1) < 40:
                return None
            return (x1, y1, x2, y2)

        rois = []
        r = _roi_from_wrist_elbow(15, 13)  # left
        if r: rois.append(r)
        r = _roi_from_wrist_elbow(16, 14)  # right
        if r: rois.append(r)

        best_label, best_conf = None, None

        for (x1, y1, x2, y2) in rois:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_rgb = cv2.resize(crop_rgb, (input_size, input_size),
                                  interpolation=cv2.INTER_LINEAR)

            crop_hands = hands.process(crop_rgb)
            if not crop_hands.multi_hand_landmarks:
                continue

            hand_lm = crop_hands.multi_hand_landmarks[0]

            # (1) HAND-SIZE GATE (skip tiny/noisy hands)
            area = hand_area_from_landmarks(hand_lm)
            if area < MIN_HAND_AREA:
                continue

            feats = extract_hand_features(hand_lm).reshape(1, -1)

            probs = hand_sign_model.predict_proba(feats)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = hand_sign_le.inverse_transform([idx])[0]

            # (2) CONFIDENCE GATE (must pass BOTH)
            if conf < HAND_SIGN_CONFIDENCE_THRESHOLD or conf < MIN_SIGN_CONF:
                continue

            # keep only your explicit cheating labels
            if label in NORMAL_HAND_LABELS or label not in CHEATING_HAND_LABELS:
                continue

            if best_conf is None or conf > best_conf:
                best_label, best_conf = label, conf

        # Nothing strong enough this frame => reset stability
        if best_label is None:
            self._prev_sign_label = None
            self._stable_sign_count = 0
            return None, None

        # (3) STABILITY GATE
        if best_label == self._prev_sign_label:
            self._stable_sign_count += 1
        else:
            self._prev_sign_label = best_label
            self._stable_sign_count = 1

        if self._stable_sign_count >= STABLE_FRAMES:
            return best_label, best_conf

        return None, None

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
                # if not self.extra_person_ever_seen:
                #     self.extra_person_ever_seen = True
                #     self.extra_person_initial_evidence_file = self.save_evidence_frame(
                #         frame, ["Initial extra person detection"], prefix="extra_person_initial"
                #     )
                #
                #     print("[INFO] Extra person first seen, initial evidence saved.")

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
            violation_types.add("pose_not_detected")#can't see the body due to lighting or unclear pose or person left camera
            violation_messages.append("Student not visible (pose not detected)")

        else:
            pose_landmarks= pose_results.pose_landmarks.landmark

            ok_vis, msg_vis = self.check_head_and_hands_visible(pose_landmarks)
            if not ok_vis:
                violation_types.add("out_of_frame")
                violation_messages.append(msg_vis)


            ok_lean, msg_lean = self.check_leaning(pose_landmarks)
            if not ok_lean:
                violation_types.add("leaning")
                violation_messages.append(msg_lean)

            ok_hand, msg_hand = self.check_hand_signals(pose_landmarks)
            if not ok_hand:
                violation_types.add("hand_signals")
                violation_messages.append(msg_hand)

        if violation_types:
            self.violation_frames += 1
            self.violation_history.append((time.time(), list(violation_types)))

        return violation_types, violation_messages
    def update_events(self, violation_types, violation_messages, frame, explicit_sign=None):
        """
        Event-based evidence:
        - Each violation type is tracked independently.
        - ONE evidence screenshot per event (when it lasts >= confirm seconds).
        - New evidence only after the event ends (and cooldown passes).
        explicit_sign: (label, conf) or None
        """
        now = time.time()

        # Expand hand sign into its own event key (per label)
        # so "1 held for 5 minutes" => only 1 screenshot.
        extra_keys = set()
        if explicit_sign is not None:
            sign_label, conf = explicit_sign
            # Use per-label key so a different sign becomes a new event
            extra_keys.add(f"explicit_hand_sign:{sign_label}")

        # Build the set of "active keys" this frame
        active_keys = set(violation_types) | extra_keys

        # 1) Mark/advance active events
        for key in active_keys:
            st = self.event_states.get(key)

            # Determine base type for thresholds
            base_type = key.split(":")[0]

            confirm_s = self.event_confirm_seconds.get(base_type, 2.0)

            # Per-sign confirm time for explicit hand signs
            if base_type == "explicit_hand_sign":
                sign_label = key.split(":", 1)[1]
                if sign_label in {"1", "2", "3", "4"}:
                    confirm_s = 0.0  # instant
                else:
                    confirm_s = 1.0  # example delay for other signs (tune)

            if st is None:
                # start new event
                self.event_states[key] = {
                    "active": True,
                    "start": now,
                    "evidence_taken": False,
                    "last_seen": now,
                }
                st = self.event_states[key]
                # beep once at event start (optional)
                play_beep()
            else:
                # continue existing event
                st["active"] = True
                st["last_seen"] = now

            # Take evidence ONCE when it passes confirm time
            elapsed = now - st["start"]
            if (not st["evidence_taken"]) and (elapsed >= confirm_s):
                self.warning_count += 1

                # Make prefix readable & unique
                safe_key = key.replace(":", "_")
                msgs = violation_messages.copy()

                # If it's a hand sign key, attach a clear message
                if base_type == "explicit_hand_sign":
                    sign_label = key.split(":", 1)[1]
                    msgs = [f"Recognized hand sign '{sign_label}'"] + msgs

                evidence_file = self.save_evidence_frame(
                    frame, msgs, prefix=safe_key
                )
                st["evidence_taken"] = True

                self.event_log.append({
                    "time": now,
                    "key": key,
                    "duration": round(elapsed, 2),
                    "file": evidence_file,
                    "messages": msgs
                })

        # 2) Close events that disappeared (end of event)
        for key, st in list(self.event_states.items()):
            if not st.get("active"):
                continue
            if key not in active_keys:
                # event ended this frame
                st["active"] = False
                st["end"] = now
                st["cooldown_until"] = now + self.event_cooldown_seconds

        # 3) Cleanup (optional): drop old inactive events after cooldown
        for key, st in list(self.event_states.items()):
            if st.get("active"):
                continue
            if st.get("cooldown_until", 0) <= now:
                # allow future fresh events; remove state
                self.event_states.pop(key, None)

    # def update_warnings(self, violation_types, violation_messages, frame):
    #     now = time.time()
    #
    #     if violation_types:
    #         if not self.current_violation_types:
    #             self.current_violation_start = now
    #             self.current_violation_types = violation_types.copy()
    #             self.current_violation_messages = violation_messages.copy()
    #             play_beep()
    #         else:
    #             self.current_violation_types |= violation_types
    #             for m in violation_messages:
    #                 if m not in self.current_violation_messages:
    #                     self.current_violation_messages.append(m)
    #
    #             elapsed = now - self.current_violation_start
    #             if elapsed >= self.warning_delay:
    #                 self.warning_count += 1
    #                 # build a readable prefix from the violation types
    #                 prefix = "_".join(
    #                     sorted(self.current_violation_types)) if self.current_violation_types else "evidence"
    #                 evidence_file = self.save_evidence_frame(frame, self.current_violation_messages, prefix=prefix)
    #
    #                 self.warning_events.append({
    #                     "time": now,
    #                     "types": self.current_violation_types.copy(),
    #                     "messages": self.current_violation_messages.copy(),
    #                     "file": evidence_file
    #                 })
    #                 print(f"[WARN] Warning #{self.warning_count} registered.")
    #                 play_beep()
    #                 self.current_violation_start = now
    #     else:
    #         self.current_violation_types = set()
    #         self.current_violation_messages = []
    #         self.current_violation_start = None

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
    evidence_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")
    os.makedirs(evidence_dir, exist_ok=True)

    detector = CheatingDetector(evidence_dir)

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
                # hand_raised_for_signal = detector.is_any_hand_raised(lm_pose)

            if (
                # hand_sign_model is not None and
                # results_hands.multi_hand_landmarks and
                hand_sign_model is not None and hand_raised_for_signal  # only classify if hand is raised
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

            #detector.update_warnings(violation_types, violation_messages, frame)
            detector.update_events(violation_types, violation_messages, frame, explicit_sign=None)

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