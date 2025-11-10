import cv2
from ultralytics import YOLO
from pathlib import Path

# Get the path to this script's directory
BASE_DIR = Path(__file__).resolve().parent

# Load the trained YOLOv8 model relative to the script location
mdel_path = BASE_DIR / "results/duckieBotSlot_pat50/weights/best.pt"
print(f"Loading model from: {mdel_path}")
model = YOLO(mdel_path)

# Open default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, imgsz=640, verbose=False)

    # Draw results on frame
    annotated_frame = results[0].plot()

    # Show annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Break with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
