import cv2
import numpy as np
from ultralytics.utils.ops import xywhr2xyxyxyxy


def iou(bbox_a, bbox_b):
    """Calculate polygon IoU for two (cx, cy, width, height, theta) OBBs."""
    boxes = np.asarray([bbox_a, bbox_b], dtype=np.float32)
    corners_a, corners_b = xywhr2xyxyxyxy(boxes)

    area_a = abs(cv2.contourArea(corners_a))
    area_b = abs(cv2.contourArea(corners_b))
    intersection, _ = cv2.intersectConvexConvex(corners_a, corners_b)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def center_error(bbox_a, bbox_b):
    """Calculate Euclidean center distance in pixels for two OBBs."""
    center_a = np.asarray(bbox_a[:2], dtype=np.float32)
    center_b = np.asarray(bbox_b[:2], dtype=np.float32)
    return float(np.linalg.norm(center_a - center_b))
