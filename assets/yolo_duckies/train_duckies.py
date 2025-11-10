from ultralytics import YOLO
import os

# Check if the current working directory is the same as the script's directory
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    # If not, change the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load a model
model = YOLO("yolov8n.pt")  # load an official model, here nano (smallest but fastest model)

# Train the model
model.train(
    data=os.path.join(os.getcwd(), "data/data.yaml"),
    epochs=30,
    imgsz=640,
    project="results/duckies",
    name="duckie-train"
)