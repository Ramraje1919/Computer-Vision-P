
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_PATH = Path("outputs/first_frame.jpg")
OUTPUT_PATH = Path("outputs/detection_preview.jpg")

CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 960
NMS_IOU_THRESHOLD = 0.50


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}. "
            "Run python src/inspect_video.py first."
        )

    print("Loading pretrained YOLO model...")
    model = YOLO("yolo11n.pt")

    print("Detecting people...")

    results = model.predict(
        source=str(IMAGE_PATH),
        classes=[0],
        conf=CONFIDENCE_THRESHOLD,
        imgsz=IMAGE_SIZE,
        iou=NMS_IOU_THRESHOLD,
        max_det=300,
        verbose=False
    )

    result = results[0]
    annotated_image = result.orig_img.copy()

    boxes = result.boxes
    person_count = len(boxes) if boxes is not None else 0

    # Draw compact labels: P 0.85 instead of person 0.85
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].cpu().tolist()
            )
            confidence = float(box.conf[0].cpu())

            label = f"P {confidence:.2f}"

            # Bounding box
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # Compact label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            label_y = max(y1, text_h + baseline + 4)

            cv2.rectangle(
                annotated_image,
                (x1, label_y - text_h - baseline - 4),
                (x1 + text_w + 6, label_y),
                (255, 0, 0),
                -1
            )

            cv2.putText(
                annotated_image,
                label,
                (x1 + 3, label_y - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), annotated_image):
        raise RuntimeError(f"Could not save: {OUTPUT_PATH}")

    print("-" * 40)
    print("PERSON DETECTION COMPLETED")
    print(f"People detected: {person_count}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Output: {OUTPUT_PATH.resolve()}")
    print("-" * 40)


if __name__ == "__main__":
    main()
