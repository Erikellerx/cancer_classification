import torch
from torchvision.models import swin_v2_b


def get_swin_tf(num_classes=2):
    """
    Returns a Swin Transformer model with a modified head for binary classification.
    """
    model = swin_v2_b(weights='DEFAULT')
    model.head = torch.nn.Linear(model.head.in_features, 2)
    return model