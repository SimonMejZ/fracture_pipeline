import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


def build_classifier(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """ResNet18 pretrained on ImageNet, fine-tuned as a fracture/no-fracture
    classifier. Freezing the backbone keeps training feasible."""
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_segmenter(num_classes: int = 2) -> nn.Module:
    """DeepLabV3-ResNet50 pretrained on COCO, fine-tuned for binary
    fracture-line segmentation."""
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier = None
    return model
