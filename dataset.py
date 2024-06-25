import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class LeafDisease(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))

        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path).convert("L"))

        mask = (mask == 38).astype(np.float32)

        if self.transform is not None:
            transformed = self.transform(image = image, mask = mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
        





