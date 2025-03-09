import os
from PIL import Image
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Path to the 'data' directory. E.g. 'CANCER/data'
            split (str): Which split to load ('train' or 'test')
            transform (callable, optional): Optional transform to be applied
                on an image sample (e.g., torchvision transforms).
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Classes in subdirectories; if you have more classes, just add them here.
        self.classes = ['Benign', 'Malignant']
        
        # Collect all image paths and their corresponding labels
        self.image_paths = []
        self.labels = []
        
        # For each class in self.classes, gather all image paths
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, self.split, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Open and convert to RGB (to ensure consistent channel ordering)
        image = Image.open(img_path).convert('RGB')
        
        # Apply any transforms (e.g., resizing, normalization, etc.)
        if self.transform:
            image = self.transform(image)
        
        return image, label