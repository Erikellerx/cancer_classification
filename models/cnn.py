import torch
from torchvision.models import resnext101_64x4d
from torchvision.models import convnext_large
from torchvision.models import efficientnet_v2_l




def get_resnext101_64x4d(num_classes):
    model = resnext101_64x4d(weights='DEFAULT')
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def get_convnext_large(num_classes):
    model = convnext_large(weights='DEFAULT')
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def get_efficientnet_v2_l(num_classes):
    model = efficientnet_v2_l(weights='DEFAULT')
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model
    