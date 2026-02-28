import cv2
import supervision as sv
from ultralytics import YOLO

# ----------------------
# CONFIG
# ----------------------
MODEL_PATH = r"C:\Users\user\Desktop\Football Project\src\yolov8m.pt"
VIDEO_PATH = r"C:\Users\user\Desktop\Football Project\data\FULL MATCH _ Belgium 3-0 Russia _ VIP Tactical Camera _ EURO 2020 _.mp4"

# ----------------------
# LOAD MODEL
# ----------------------
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("❌ Cannot open video")

# ----------------------
# ANNOTATORS
# ----------------------
player_box = sv.BoxAnnotator(color=sv.Color.RED, thickness=2)
player_label = sv.LabelAnnotator(color=sv.Color.RED)

ref_box = sv.BoxAnnotator(color=sv.Color.BLUE, thickness=2)
ref_label = sv.LabelAnnotator(color=sv.Color.BLUE)

ball_box = sv.BoxAnnotator(color=sv.Color.WHITE, thickness=2)
ball_label = sv.LabelAnnotator(color=sv.Color.WHITE)

print("▶ Testing started — Press Q to quit")

# ----------------------
# LOOP
# ----------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.1,
        iou=0.5,
        imgsz=640,
        verbose=False
    )[0]

    detections = sv.Detections.from_ultralytics(results)

    if detections.tracker_id is None:
        cv2.imshow("YOLO Football Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    # ----------------------
    # SPLIT CLASSES
    # ----------------------
    players = detections[detections.class_id == 0]
    referees = detections[detections.class_id == 1]
    balls = detections[detections.class_id == 2]

    # ----------------------
    # LABELS
    # ----------------------
    player_labels = [f"P{tid}" for tid in players.tracker_id]
    ref_labels = [f"REF {tid}" for tid in referees.tracker_id]
    ball_labels = ["BALL"] * len(balls)

    # ----------------------
    # DRAW
    # ----------------------
    frame = player_box.annotate(frame, players)
    frame = player_label.annotate(frame, players, player_labels)

    frame = ref_box.annotate(frame, referees)
    frame = ref_label.annotate(frame, referees, ref_labels)

    frame = ball_box.annotate(frame, balls)
    frame = ball_label.annotate(frame, balls, ball_labels)

    cv2.imshow("YOLO Football Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
