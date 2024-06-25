import os
import random
import shutil

def split_dataset(img_dir, mask_dir, train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, split_ratio=0.8):
    # List all image files
    images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    # Shuffle the images to ensure random split
    random.shuffle(images)
    
    # Calculate the number of training images
    num_train = int(len(images) * split_ratio)
    
    # Split the images into training and validation sets
    train_images = images[:num_train]
    val_images = images[num_train:]
    
    # Copy images and masks to respective directories
    for img in train_images:
        img_name = os.path.basename(img)
        mask_name = img_name.replace('.jpg', '.png')
        
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(train_img_dir, img_name))
        shutil.copy(os.path.join(mask_dir, mask_name), os.path.join(train_mask_dir, mask_name))
    
    for img in val_images:
        img_name = os.path.basename(img)
        mask_name = img_name.replace('.jpg', '.png')
        
        shutil.copy(os.path.join(img_dir, img_name), os.path.join(val_img_dir, img_name))
        shutil.copy(os.path.join(mask_dir, mask_name), os.path.join(val_mask_dir, mask_name))

# Example usage:
image_directory = 'aug_data/images'
mask_directory = 'aug_data/masks'

train_img_directory = 'dataset/train_imgs'
train_mask_directory = 'dataset/train_masks'
val_img_directory = 'dataset/val_images'
val_mask_directory = 'dataset/val_masks'

os.makedirs(train_img_directory, exist_ok=True)
os.makedirs(train_mask_directory, exist_ok=True)
os.makedirs(val_img_directory, exist_ok=True)
os.makedirs(val_mask_directory, exist_ok=True)

split_dataset(image_directory, mask_directory, train_img_directory, train_mask_directory, val_img_directory, val_mask_directory, split_ratio=0.8)

