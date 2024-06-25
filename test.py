import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.unet import UNET 
import os
import albumentations as A

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


transform = Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_model(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image = augmented['image']
    return image.unsqueeze(0)


def load_ground_truth_mask(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask == 38).astype(np.uint8)
    return mask


def perform_inference(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = (output > 0.5).float()
        return output.cpu().numpy()


model = UNET(3, 1).to(DEVICE)
model = load_model('checkpoints/checkpoints_50.pt', model)

img_list = os.listdir('dataset/val_images')
save_path = 'results/'
ground_truth_mask_path = 'dataset/val_masks/'

for img in img_list:

    image_path = 'dataset/val_images/' + img  
    image_tensor = preprocess_image(image_path)


    output_mask = perform_inference(model, image_tensor)
    output_mask = output_mask.squeeze()  

    original_image = cv2.imread(image_path)[:, :, ::-1]
    resized_mask = cv2.resize(output_mask, (original_image.shape[1], original_image.shape[0]))


    ground_truth_mask = load_ground_truth_mask(os.path.join(ground_truth_mask_path, img.replace('.jpg', '.png')))
    resized_ground_truth_mask = cv2.resize(ground_truth_mask, (original_image.shape[1], original_image.shape[0]))


    gt_overlay = np.zeros_like(original_image)
    gt_overlay[resized_ground_truth_mask > 0] = [0, 0, 255]

    masked_image_gt = original_image.copy()
    masked_image_gt[resized_ground_truth_mask > 0] = (masked_image_gt[resized_ground_truth_mask > 0] * 0.5 + 
                                                     gt_overlay[resized_ground_truth_mask > 0] * 0.5).astype(np.uint8)


    overlay = np.zeros_like(original_image)
    overlay[resized_mask > 0.5] = [0, 0, 255]  


    masked_image_pred = original_image.copy()
    masked_image_pred[resized_mask > 0.5] = (masked_image_pred[resized_mask > 0.5] * 0.5 + 
                                             overlay[resized_mask > 0.5] * 0.5).astype(np.uint8)


    concatenated_image = np.concatenate((masked_image_gt, masked_image_pred), axis=1)

    cv2.imwrite(os.path.join(save_path, img), concatenated_image)
