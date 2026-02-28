import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, deque
import torch

# ----------------------
# DEVICE SETUP
# ----------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device.upper()}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA not available, using CPU")

model = YOLO(r"C:\Users\user\Desktop\Football Project\src\yolov8m.pt")
model.to(device)

video_path = r"C:\Users\user\Desktop\Football Project\data\مباراة الكلاسيكو برشلونة وريال مدريد 3-2 الدوري الاسباني (شاشة كاملة ) تعليق الع.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Cannot open video")

# ----------------------
# MEMORY STRUCTURES
# ----------------------
player_team_map = {}
player_team_history = defaultdict(lambda: deque(maxlen=30))
referees = set()
ball_history = deque(maxlen=5)

# ----------------------
# VISUAL COLORS (BGR)
# ----------------------
TEAM_COLORS = {
    0: (255, 0, 0),        # BLUE
    1: (255, 255, 255)     # WHITE
}

REFEREE_COLOR = (0, 255, 255)  # YELLOW
BALL_COLOR = (0, 255, 0)       # GREEN

frame_count = 0
RECLASSIFY_INTERVAL = 30

print("▶ Tracking started")
print("Team A: BLUE jerseys")
print("Team B: WHITE jerseys")
print("Referee: YELLOW jersey")
print("Press Q to quit")

# ----------------------
# HELPERS
# ----------------------

def get_dominant_color_hsv(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h = y2 - y1
    w = x2 - x1

    roi = frame[y1 + int(h * 0.05):y1 + int(h * 0.4),
                x1 + int(w * 0.2):x2 - int(w * 0.2)]

    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    pixels = pixels[pixels[:, 2] > 40]
    if len(pixels) == 0:
        return None

    return np.median(pixels, axis=0)


def classify_color(hsv_color):
    """
    BLUE -> Team A
    WHITE -> Team B
    YELLOW -> Referee
    """
    if hsv_color is None:
        return "UNKNOWN"

    h, s, v = hsv_color

    # BLUE
    if 90 < h < 130 and s > 70 and v > 60:
        return "BLUE"

    # WHITE
    if s < 40 and v > 180:
        return "WHITE"

    # YELLOW (wider range for stability)
    if 18 < h < 40 and s > 70 and v > 70:
        return "YELLOW"

    return "UNKNOWN"


def majority_vote(history):
    if not history:
        return None
    return max(set(history), key=history.count)


def is_valid_ball(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    area = w * h

    if area < 50 or area > 2000:
        return False

    ratio = w / h if h > 0 else 0
    if ratio < 0.6 or ratio > 1.4:
        return False

    if y1 < frame_shape[0] * 0.33:
        return False

    return True


# ----------------------
# MAIN LOOP
# ----------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.track(
        frame,
        imgsz=640,
        conf=0.3,
        iou=0.5,
        persist=True,
        classes=[0, 32],
        tracker="botsort.yaml",
        verbose=False,
        device=device
    )[0]

    detections = sv.Detections.from_ultralytics(results)

    if detections.tracker_id is None:
        cv2.imshow("Football Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    players = detections[detections.class_id == 0]
    balls = detections[detections.class_id == 32]

    should_reclassify = frame_count % RECLASSIFY_INTERVAL == 0

    # ----------------------
    # COLOR CLASSIFICATION
    # ----------------------

    for bbox, tid in zip(players.xyxy, players.tracker_id):

        if tid in referees and not should_reclassify:
            continue

        if tid in player_team_map and len(player_team_history[tid]) >= 20 and not should_reclassify:
            continue

        hsv_color = get_dominant_color_hsv(frame, bbox)
        color_class = classify_color(hsv_color)

        if color_class == "BLUE":
            player_team_history[tid].append(0)
            referees.discard(tid)

        elif color_class == "WHITE":
            player_team_history[tid].append(1)
            referees.discard(tid)

        elif color_class == "YELLOW":
            referees.add(tid)
            player_team_map.pop(tid, None)

        if tid not in referees and len(player_team_history[tid]) >= 5:
            stable = majority_vote(player_team_history[tid])
            if stable is not None:
                player_team_map[tid] = stable

    # ----------------------
    # KMEANS FALLBACK
    # ----------------------

    if frame_count == 60:

        features = []
        ids = []

        for bbox, tid in zip(players.xyxy, players.tracker_id):
            if tid not in referees and tid not in player_team_map:
                hsv = get_dominant_color_hsv(frame, bbox)
                if hsv is not None:
                    features.append(hsv)
                    ids.append(tid)

        if len(features) >= 6:
            kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
            labels = kmeans.fit_predict(features)

            cluster_assignments = {}

            for cluster_id in range(3):
                cluster_colors = []
                for i, label in enumerate(labels):
                    if label == cluster_id:
                        cluster_colors.append(classify_color(features[i]))
                cluster_assignments[cluster_id] = cluster_colors

            cluster_to_team = {}

            for cid, colors in cluster_assignments.items():
                if colors.count("BLUE") > len(colors) * 0.5:
                    cluster_to_team[cid] = 0
                elif colors.count("WHITE") > len(colors) * 0.5:
                    cluster_to_team[cid] = 1
                elif colors.count("YELLOW") > len(colors) * 0.3:
                    cluster_to_team[cid] = "REF"

            for tid, label in zip(ids, labels):
                assign = cluster_to_team.get(label)
                if assign == "REF":
                    referees.add(tid)
                elif assign is not None:
                    player_team_history[tid].append(assign)
                    player_team_map[tid] = assign

    # ----------------------
    # BALL FILTER
    # ----------------------

    valid_ball = None
    for i, bbox in enumerate(balls.xyxy):
        if is_valid_ball(bbox, frame.shape):
            conf = balls.confidence[i]
            if valid_ball is None or conf > valid_ball[1]:
                valid_ball = (bbox, conf)

    if valid_ball:
        ball_history.append(valid_ball[0])

    # ----------------------
    # DRAW
    # ----------------------

    for bbox, tid in zip(players.xyxy, players.tracker_id):
        x1, y1, x2, y2 = map(int, bbox)

        if tid in referees:
            color = REFEREE_COLOR
            label = f"REF | {tid}"
        else:
            team = player_team_map.get(tid, 0)
            color = TEAM_COLORS[team]
            team_name = "BLUE" if team == 0 else "WHITE"
            label = f"{team_name} | {tid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if len(ball_history) > 0:
        x1, y1, x2, y2 = map(int, ball_history[-1])
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = max(10, (x2 - x1) // 2)
        cv2.circle(frame, center, radius, BALL_COLOR, -1)

    cv2.imshow("Football Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()