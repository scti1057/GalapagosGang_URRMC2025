import os
import cv2

from ultralytics import YOLO

# Check if the current working directory is the same as the script's directory
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    # If not, change the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model
model_best  = YOLO(os.path.join(os.getcwd(), "results/duckieBotSlot/weights/best.pt"))   # load the best model
model_last  = YOLO(os.path.join(os.getcwd(), "results/duckieBotSlot/weights/last.pt"))   # load the latest model

# Run inference on an image
picName = "woDuckies.png"
img_path = os.path.join(os.getcwd(), "testPics", picName)
results = model_best(
    img_path,  # predict on an image
    save=True,
    project="results/predictions",
    name=picName
)

# Load the image for drawing
img = cv2.imread(img_path)

# Loop over detected results
cls_id = None
conf = None
xyxy = None
for result in results:
    boxes = result.boxes  # bounding boxes object

    for box in boxes:
        cls_id  = int(box.cls)           # class ID
        conf    = float(box.conf)          # confidence score
        xyxy    = box.xyxy.tolist()[0]     # box coordinates [x1, y1, x2, y2]

        print(f"Detected class {cls_id} with confidence {conf:.2f} at {xyxy}")

        if cls_id is not None and conf is not None:
            # Print the detected class name 
            label = model_best.names[cls_id]
            print(f"Detected {label} with confidence {conf:.2f}")
            # Draw bounding box and label on the image
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Show the image with bounding boxes and labels
cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()