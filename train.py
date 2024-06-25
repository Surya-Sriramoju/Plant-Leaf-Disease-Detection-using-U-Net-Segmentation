import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model.unet import UNET
import multiprocessing
from dataset import LeafDisease
from torch.utils.data import DataLoader
from utils import mIOU, evaluate, precision_recall_f1, DiceLoss, CombinedLoss
import os
import matplotlib.pyplot as plt

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 100
NUM_WORKERS = multiprocessing.cpu_count()//4
PIN_MEMORY = True
LOAD_MODEL = True

train_img = 'dataset/train_imgs'
train_mask = 'dataset/train_masks'
val_img = 'dataset/val_images'
val_masks = 'dataset/val_masks'


def train_fn(loader, model, optimizer, loss_fn, scaler = 0):
    x = 0
    model.train()
    curr_loss = 0.0
    curr_miou = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for batch in tqdm(loader):
        image, labels = batch
        image, labels = image.to(DEVICE), labels.float().unsqueeze(1).to(DEVICE)
        prediction = model(image)
        # debug
        # print(prediction.shape)
        optimizer.zero_grad()
        loss = loss_fn(prediction, labels)
        loss.backward()
        optimizer.step()
        miou = mIOU(prediction, labels)
        precision, recall, f1 = precision_recall_f1(prediction, labels)
        curr_loss = curr_loss + loss.item()*image.size(0)
        curr_miou = curr_miou + miou*image.size(0)
        total_precision += precision*image.size(0)
        total_recall += recall*image.size(0)
        total_f1 += f1*image.size(0)
    num_samples = len(loader.dataset)
    return curr_loss/num_samples, curr_miou/num_samples, total_precision / num_samples, total_recall / num_samples, total_f1 / num_samples
        

def start_training(loader, model, optimizer, loss_fn, scaler, val_loader, scheduler):
    train_loss_list = []
    running_miou_list = []
    train_f1_list = []
    val_loss_list = []
    val_miou_list = []
    val_f1_list = []
    save_path = 'checkpoints'
    patience = 10
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1,NUM_EPOCHS+1):
        torch.cuda.empty_cache()
        model.train()
        print("Epoch "+str(epoch))
        train_loss, train_miou, train_precision, train_recall, train_f1 = train_fn(loader, model, optimizer, loss_fn, 0)
        train_loss_list.append(train_loss)
        running_miou_list.append(train_miou)
        train_f1_list.append(train_f1)
        val_loss, val_mIOU, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device = DEVICE, loss_function=loss_fn)
        val_loss_list.append(val_loss)
        val_miou_list.append(val_mIOU)
        val_f1_list.append(val_f1)
        if scheduler is not None:
            scheduler.step(val_loss)
        print(f'Epoch [{epoch}/{NUM_EPOCHS}], Train Loss: {train_loss:.4f}, Train IOU: {train_f1:.4f}, Train F1: {train_miou:.4f},Val Loss: {val_loss:.4f}, Val IOU: {val_mIOU:.4f},Val F1: {val_f1:.4f}')
        if epoch%10 == 0:
            save_checkpoint(save_path=save_path, model=model, optimizer=optimizer, val_loss=0, epoch=epoch)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            save_checkpoint(save_path=save_path, model=model, optimizer=optimizer, val_loss=0, epoch=epoch)
            print("Early stopping")
            break

    return train_loss_list, running_miou_list, train_f1_list, val_loss_list, val_miou_list, val_f1_list

        



def save_checkpoint(save_path, model, optimizer, val_loss, epoch):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    file_name = save_path.split("/")[-1].split("_")[0] + "_" + str(epoch) + ".pt"
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, os.path.join(save_path, file_name))

def draw_plot(train,val, flag):
    if flag == 'loss':
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, 'b', label='Training Loss')
        plt.plot(epochs, val, 'r', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.savefig('loss.png')
    if flag == 'iou':
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, 'b', label='Training iou')
        plt.plot(epochs, val, 'r', label='Validation iou')
        plt.title('Training and Validation IOU')
        plt.xlabel('Epochs')
        plt.ylabel('IOU')
        plt.savefig('iou.png')
    if flag == 'f1':
        epochs = range(1, len(train) + 1)
        plt.plot(epochs, train, 'b', label='Training F1')
        plt.plot(epochs, val, 'r', label='Validation F1')
        plt.title('Training and Validation f1')
        plt.xlabel('Epochs')
        plt.ylabel('F1')
        plt.savefig('F1_Score.png')


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0



def main():
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomRotate90(),
        A.Transpose(),
        A.OneOf([
            A.RandomResizedCrop(256, 256, scale=(0.8, 1.0)),
        ], p=0.5),
        A.ElasticTransform(),
        A.HueSaturationValue(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])



    train_ds = LeafDisease(train_img, train_mask, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory= PIN_MEMORY, shuffle=True)
    val_ds = LeafDisease(val_img, val_masks, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory= PIN_MEMORY, shuffle=False)


    model = UNET(3, 1).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = CombinedLoss(dice_weight=0.5, bce_weight=0.5, smooth=1e-6)
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.2, patience=2)
    # train_fn(train_loader, model, optimizer, loss_fn, 0)
    train_loss_list, running_miou_list, train_f1_list, val_loss_list, val_miou_list, val_f1_list = start_training(train_loader, model, optimizer, loss_fn, 0, val_loader, scheduler)

    draw_plot(train_loss_list, val_loss_list, 'loss')
    draw_plot(running_miou_list, val_miou_list, 'iou')
    draw_plot(train_f1_list, val_f1_list, 'f1')




if __name__ == "__main__":
    main()