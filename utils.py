import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


def mIOU(preds, labels, smooth=1e-6):
    preds = preds.squeeze(1)
    labels = labels.squeeze(1)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    intersection = (preds * labels).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + labels.sum(dim=(1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def precision_recall_f1(preds, labels):
    preds = preds.squeeze(1)
    labels = labels.squeeze(1)
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()
    
    tp = (preds * labels).sum(dim=(1, 2))
    fp = ((1 - labels) * preds).sum(dim=(1, 2))
    fn = (labels * (1 - preds)).sum(dim=(1, 2))
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision.mean().item(), recall.mean().item(), f1.mean().item()

def evaluate(model, data_loader, device, loss_function):
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    with torch.no_grad():
        model.eval()
        for image, labels in data_loader:
            image, labels = image.to(device), labels.unsqueeze(1).to(device)
            prediction = model(image)
            loss = loss_function(prediction, labels)
            miou = mIOU(prediction, labels)
            precision, recall, f1 = precision_recall_f1(prediction, labels)
            total_loss += loss.item() * image.size(0)
            total_miou += miou* image.size(0)
            total_precision += precision* image.size(0)
            total_recall += recall* image.size(0)
            total_f1 += f1* image.size(0)
            num_samples = len(data_loader.dataset)
    
    return (total_loss / num_samples, total_miou / num_samples,
            total_precision / num_samples, total_recall / num_samples, total_f1 / num_samples)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, labels):
        preds = torch.sigmoid(preds)
        intersection = (preds * labels).sum()
        dice_score = (2. * intersection + self.smooth) / (preds.sum() + labels.sum() + self.smooth)
        return 1 - dice_score
    
class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, preds, labels):
        dice_loss = self.dice_loss(preds, labels)
        bce_loss = self.bce_loss(preds, labels)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

            
