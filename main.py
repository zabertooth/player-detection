from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("best.pt")

# Open video
cap = cv2.VideoCapture("15sec_input_720p.mp4")

# Setup output video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("tracked_output.avi", cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# DeepSort tracker
tracker = DeepSort(max_age=15, n_init=2)

# Class index for players
PLAYER_CLASS_ID = 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect
    results = model(frame)[0]
    boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    # Collect player detections only
    detections = []
    for box, cls, conf in zip(boxes, classes, confs):
        if int(cls) == PLAYER_CLASS_ID:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, 'player'))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        l, t, r, b = track.to_ltrb()
        track_id = track.track_id
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (int(l), int(t) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
