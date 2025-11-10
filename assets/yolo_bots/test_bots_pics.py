import os

from ultralytics import YOLO

# Check if the current working directory is the same as the script's directory
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    # If not, change the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the trained model
model_best  = YOLO(os.path.join(os.getcwd(), "results/bots/bots-train/weights/best.pt"))   # load the best model
model_last  = YOLO(os.path.join(os.getcwd(), "results/bots/bots-train/weights/last.pt"))   # load the latest model

# Run inference on an image
picName = "wBotNear.png"
results = model_best(
    os.path.join(os.getcwd(), "testPics", picName),  # predict on an image
    save=True,
    project="results/predictions",
    name=picName
)

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