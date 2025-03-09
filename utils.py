import glob
import os
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from models.cnn import get_resnext101_64x4d, get_convnext_large, get_efficientnet_v2_l
from models.transformers import get_swin_tf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_images_glob(folder_path):
    images = []
    for file_path in tqdm(glob.glob(os.path.join(folder_path, "*.*"))):
        with Image.open(file_path) as img:
            # Convert to RGB just to standardize
            images.append(img.convert("RGB"))
    return images


def get_model(name = "resnext"):
    if name == "resnext":
        return get_resnext101_64x4d(num_classes=2)
    elif name == "convnext":
        return get_convnext_large(num_classes=2)
    elif name == "efficientnet":
        return get_efficientnet_v2_l(num_classes=2)
    elif name == "swin":
        return get_swin_tf(num_classes=2)
    else:
        raise ValueError(f"Model {name} is not supported.")