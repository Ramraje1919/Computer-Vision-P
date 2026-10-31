
import cv2
from pathlib import Path

video_path = Path("data/MOT16-03.mp4")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Video loaded successfully!")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {frame_count}")

if fps > 0:
    print(f"Duration: {frame_count / fps:.2f} seconds")

success, frame = cap.read()

if success:
    cv2.imwrite("outputs/first_frame.jpg", frame)
    print("First frame saved to outputs/first_frame.jpg")

cap.release()
