from ultralytics import YOLO
import os

# Check if the current working directory is the same as the script's directory
if os.getcwd() != os.path.dirname(os.path.abspath(__file__)):
    # If not, change the current working directory to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

myPatience = 50
myDevice = -1       # -1 for cpu; 0 for first gpu


if __name__ == "__main__":
    # Load a model
    model = YOLO(os.path.join(os.getcwd(), "yolov8n.pt"))  # load an official model, here nano (smallest but fastest model)

    # Train the model
    model.train(
        data=os.path.join(os.getcwd(), "data/data.yaml"),
        epochs=2000,  # Number of epochs to train for
        patience=myPatience,  # Early stopping patience
        imgsz=640,  # Image size for training
        project="results",  # Project name for saving results
        name=f"duckieBotSlot_pat{myPatience}",  # Name of the experiment
        batch=16,   # Batch size for training
        device=myDevice,  # Use the first GPU (0) for training
        save_period=0,  # Save model every epoch
        save=True,  # Save the model after training
        exist_ok=True,  # Overwrite existing results folder
        warmup_epochs=3,  # Number of warmup epochs
    )