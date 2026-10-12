from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-obb.pt")


def detector(images):
    """Return one YOLO Results object per input image.

    Each result retains the OBB boxes, corners, classes, and confidence scores
    for its image, including empty predictions when no objects are detected.
    """
    results = model(images)
    return results



