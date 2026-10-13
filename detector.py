from ultralytics import YOLO

# Load a model and weight
model = YOLO("yolo26x-obb.pt")

def detector(images):
    """Return one YOLO Results object per input image."""
    results = model(images)
    return results



