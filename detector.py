from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-obb.pt")


def detector(images):
    results = model(images)  # predict on an image

    for image in images:
    # Access the results
        confs = []
        for result in results:
            xywhr = result.obb.xywhr  # center-x, center-y, width, height, angle (radians)
            corners = result.obb.xyxyxyxy  # polygon format with 4-points
            names = [result.names[cls.item()] for cls in result.obb.cls.int()]  # class name of each box
            confs = result.obb.conf  # confidence score of each box

        if confs > confs:
            corners.append(corners)
        """
        # make sure the program works even without detectable object
        try: corners_with_dir = calculate_dir(corners)
            continue
        """
    return corners

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




