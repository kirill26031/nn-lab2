import os
import torch
from torch.utils.data import DataLoader
from mask_rcnn import create_mask_rcnn
from torchvision.transforms import functional as F
from PIL import Image
import numpy as np
import optuna
import csv
from train_fcn import calculate_iou, calculate_dice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIRS = {
    'train': (os.path.join(BASE_DIR, 'train/images'), os.path.join(BASE_DIR, 'train/masks')),
    'validation': (os.path.join(BASE_DIR, 'validation/images'), os.path.join(BASE_DIR, 'validation/masks')),
}

batch_size = 4
epochs = 10
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        image = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(np.array(mask), dtype=torch.uint8)

        obj_ids = torch.unique(mask)[1:]  # Видаляємо фон
        if len(obj_ids) == 0:  # Якщо немає об'єктів
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = mask == obj_ids[:, None, None]
            boxes = []
            for i in range(len(obj_ids)):
                pos = torch.nonzero(masks[i], as_tuple=True)
                xmin, xmax = torch.min(pos[1]), torch.max(pos[1])
                ymin, ymax = torch.min(pos[0]), torch.max(pos[0])
                boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(obj_ids),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return image, target


def get_dataloaders(data_dirs, batch_size):
    dataloaders = {}
    for split, (image_dir, mask_dir) in data_dirs.items():
        dataset = MaskRCNNDataset(image_dir, mask_dir)
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'),
                                        collate_fn=lambda x: tuple(zip(*x)))
    return dataloaders


def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(train_loader)


def validate_model_with_metrics(model, val_loader, device, num_classes):
    model.train()
    total_loss = 0
    all_ious, all_dices = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            model.eval()
            outputs = model(images)
            model.train()

            for output, target in zip(outputs, targets):
                pred_masks = (output['masks'] > 0.5).squeeze(1)
                true_masks = target['masks']

                # IoU and Dice
                ious = calculate_iou(pred_masks, true_masks, num_classes)
                dices = calculate_dice(pred_masks, true_masks, num_classes)

                all_ious.extend(ious)
                all_dices.extend(dices)

    mean_iou = torch.tensor(all_ious).nanmean().item()
    mean_dice = torch.tensor(all_dices).nanmean().item()
    avg_loss = total_loss / len(val_loader)

    print(f"Validation - Loss: {avg_loss:.4f}, Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
    return avg_loss, mean_iou, mean_dice


def objective(trial):
    # Parameters
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    anchor_sizes_options = [(32, 64, 128), (64, 128, 256)]
    anchor_sizes_idx = trial.suggest_int("anchor_sizes_idx", 0, len(anchor_sizes_options) - 1)
    anchor_sizes = anchor_sizes_options[anchor_sizes_idx]
    num_proposals = trial.suggest_int("num_proposals", 100, 500, step=100)

    print(f"\nStarting training with parameters:")
    print(f"Learning Rate: {lr}")
    print(f"Anchor Sizes: {anchor_sizes}")
    print(f"Number of Proposals: {num_proposals}\n")

    dataloaders = get_dataloaders(DATA_DIRS, batch_size=batch_size)
    train_loader = dataloaders['train']
    val_loader = dataloaders['validation']

    model = create_mask_rcnn(num_classes=num_classes, anchor_sizes=anchor_sizes, num_proposals=num_proposals).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Frozen backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    for epoch in range(5):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = validate_model_with_metrics(model, val_loader, device, num_classes)
        print(f"Epoch {epoch + 1}/5 (Frozen Backbone), Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Fine-tuning
    for param in model.backbone.parameters():
        param.requires_grad = True

    for epoch in range(5, epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        val_loss = validate_model_with_metrics(model, val_loader, device, num_classes)
        print(f"Epoch {epoch + 1}/{epochs} (Fine-Tuning), Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    return val_loss


def run_optuna(n_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best parameters:", study.best_params)
    print("Best validation loss:", study.best_value)
    return study


def save_results_to_csv(results, filename="optuna_results.csv"):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


study = run_optuna(n_trials=10)
