from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
import torch

def build_model(finetune: bool) -> torch.nn.Module:
    """Load COCO Faster R-CNN."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    )
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


