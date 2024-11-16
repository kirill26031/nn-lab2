import torch
import os
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from fcn import FCN
from torchvision.transforms import Compose, Resize
import numpy as np
from PIL import Image
from itertools import product
import time
import csv
#from data import get_class_distribution

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIRS = {
    'train': (os.path.join(BASE_DIR, 'train/images'), os.path.join(BASE_DIR, 'train/masks')),
    'validation': (os.path.join(BASE_DIR, 'validation/images'), os.path.join(BASE_DIR, 'validation/masks')),
}

batch_sizes = [4, 8, 16]
learning_rates = [0.001, 0.0001]
skip_connections = [1, 2, 3]
architectures = ['resnet', 'vgg']
epochs = 10

num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PNGSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image, mask = self.transform(image, mask)

        image = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).permute(2, 0, 1)  # [C, H, W]
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


def get_dataloaders(data_dirs, batch_size, transform=None, target_size=(256, 256)):
    dataloaders = {}
    for split, (image_dir, mask_dir) in data_dirs.items():
        dataset = PNGSegmentationDataset(image_dir, mask_dir, transform=transform, target_size=target_size)
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
    return dataloaders


def calculate_class_weights(class_distribution):
    total_pixels = sum(class_distribution.values())
    class_weights = {cls: total_pixels / count for cls, count in class_distribution.items()}

    # Нормалізація ваг
    total_weight = sum(class_weights.values())
    class_weights = {cls: weight / total_weight for cls, weight in class_weights.items()}

    return torch.tensor(list(class_weights.values()), dtype=torch.float32).to(device)


train_class_distribution = {
    0: 67720594331,
    1: 4390472503,
    2: 7192139908,
    3: 5756712,
    4: 1775289245
}
class_weights = calculate_class_weights(train_class_distribution)

criterion = nn.CrossEntropyLoss(weight=class_weights)


# IoU
def calculate_iou(pred, target, num_classes):
    ious = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)

    return ious


# Dice
def calculate_dice(pred, target, num_classes):
    dices = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls

        intersection = (pred_cls & target_cls).sum().item()
        total = pred_cls.sum().item() + target_cls.sum().item()

        if total == 0:
            dices.append(float('nan'))
        else:
            dices.append(2 * intersection / total)

    return dices


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10, num_classes=5):
    best_loss = float('inf')
    metrics = {}

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        print(train_loss)
        end_time = time.time()
        print(f"Epoch {epoch + 1}/{epochs} completed in {end_time - start_time:.2f} seconds")

        # Валідація
        model.eval()
        valid_loss = 0.0
        all_ious, all_dices = [], []
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                valid_loss += loss.item()
                ious = calculate_iou(outputs, y, num_classes)
                dices = calculate_dice(outputs, y, num_classes)
                all_ious.extend(ious)
                all_dices.extend(dices)

        valid_loss /= len(valid_loader)
        mean_iou = torch.tensor(all_ious).nanmean().item()
        mean_dice = torch.tensor(all_dices).nanmean().item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Valid Loss: {valid_loss:.4f}, Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")

        if valid_loss < best_loss:
            print("Saving best model...")
            torch.save(model.state_dict(), "best_model.pth")
            best_loss = valid_loss
            metrics = {"loss": valid_loss, "iou": mean_iou, "dice": mean_dice}

    return metrics


def save_results_to_csv(results, filename="results.csv"):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# Основний запуск
dataloaders = get_dataloaders(DATA_DIRS, batch_size=8, transform=None, target_size=(256, 256))
train_loader = dataloaders['train']
valid_loader = dataloaders['validation']

results = []

for batch_size, lr, skips, arch in product(batch_sizes, learning_rates, skip_connections, architectures):
    print(f"Training with batch_size={batch_size}, lr={lr}, skips={skips}, architecture={arch}")

    model = FCN(num_classes=num_classes, backbone=arch, num_skip_connections=skips, dropout_rate=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics = train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=epochs, num_classes=num_classes)

    results.append({
        "batch_size": batch_size,
        "learning_rate": lr,
        "skip_connections": skips,
        "architecture": arch,
        "loss": metrics["loss"],
        "iou": metrics["iou"],
        "dice": metrics["dice"]
    })

save_results_to_csv(results)