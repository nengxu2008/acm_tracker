from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

def build_model(finetune: bool) -> torch.nn.Module:
    """Load COCO Faster R-CNN; optionally replace its head with person/background."""
    # model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    )
    # Retain low-confidence predictions for a proper precision-recall curve.
    model.roi_heads.score_thresh = 0.0
    if finetune:
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
    return model

model = build_model(finetune=False)
model.eval()

def detector(images):
    """Run Faster R-CNN on OpenCV BGR images."""
    image_tensors = []
    for image in images:
        rgb_image = image[:, :, ::-1].copy()
        image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
        image_tensors.append(image_tensor)

    with torch.inference_mode():
        return model(image_tensors)


