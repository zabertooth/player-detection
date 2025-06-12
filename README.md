# 🧍‍♂️ Player Tracking with YOLOv8 and DeepSORT

This project detects and tracks players in a video using a YOLOv8 object detection model and DeepSORT tracking algorithm. It assigns a unique ID to each player and keeps that ID consistent even when a player goes out of the frame and comes back.

---

## 📁 Project Files

```
.
├── best.pt                  # YOLOv8 trained model (player detection)
├── 15sec_input_720p.mp4     # Input video file
├── tracked_output.avi       # Output video with tracking boxes and IDs
├── main.py                  # Python script
└── README.md                # Documentation
```

---

## ⚙️ Requirements

- Python 3.8 or higher

Install dependencies:

```bash
pip install ultralytics opencv-python deep_sort_realtime
```

---

## 🚀 How to Run

1. Place your trained YOLOv8 model in the root directory as `best.pt`.
2. Place your input video as `15sec_input_720p.mp4`.
3. Run the script:

```bash
python main.py
```

The script will:
- Detect players (class ID = 2)
- Track each player with a unique ID
- Save the annotated video as `tracked_output.avi`
- Display the video with tracking live

> Press `q` to stop early.

---

## 🧠 Notes

- Make sure your YOLO model is trained to detect players. If the class index for "player" is different, update this line in the code:
  ```python
  PLAYER_CLASS_ID = 2
  ```

- `DeepSort` keeps tracking players across frames using visual appearance + motion.

---

## 🛠 Troubleshooting

- **Video not opening**: Make sure `15sec_input_720p.mp4` exists in the same folder.
- **Model not working**: Confirm `best.pt` is a valid YOLOv8 model trained with Ultralytics.
