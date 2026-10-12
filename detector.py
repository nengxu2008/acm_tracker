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

"""
def calculate_dir(corners):
    # calculate the directions of search
    corners = corners.unsqueeze(-1)
    for i in len(corners[0]):
        right_mid = 0.5 * (corners[i][0][0] - corners[i][1][0], corners[i][0][1] - corners[i][1][1])
        left_mid = 0.5 * (corners[i][2][0] - corners[i][3][0], corners[i][2][1] - corners[i][3][1])
        dir = right_mid - left_mid
        corners[4][i] = dir
    return corners
"""

detector("/Users/piefacemac/Library/CloudStorage/OneDrive-Personal/图片/Camera Roll/2022/12/20221210_064936173_iOS.jpg")




