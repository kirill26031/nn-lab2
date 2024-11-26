import os
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIRS = {
    'train': (os.path.join(BASE_DIR, 'train/images'), os.path.join(BASE_DIR, 'train/masks')),
    'validation': (os.path.join(BASE_DIR, 'validation/images'), os.path.join(BASE_DIR, 'validation/masks')),
}

batch_size = 4
epochs = 5
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset definition
class MaskRCNNDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, resize=(256, 256)):  # resizing
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # Resize
        image = image.resize(self.resize, Image.Resampling.BILINEAR)
        mask = mask.resize(self.resize, Image.Resampling.NEAREST)

        image = np.array(image)
        mask = np.array(mask)

        image = torch.tensor(np.array(image) / 255.0, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(np.array(mask), dtype=torch.uint8)

        obj_ids = torch.unique(mask)[1:]  # Remove the background
        if len(obj_ids) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = mask == obj_ids[:, None, None]
            boxes = []
            valid_obj_ids = []
            for i in range(len(obj_ids)):
                pos = torch.nonzero(masks[i], as_tuple=True)
                xmin, xmax = torch.min(pos[1]), torch.max(pos[1])
                ymin, ymax = torch.min(pos[0]), torch.max(pos[0])
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin.item(), ymin.item(), xmax.item(), ymax.item()])
                    valid_obj_ids.append(obj_ids[i])
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                masks = torch.zeros((0, mask.shape[0], mask.shape[1]), dtype=torch.uint8)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                masks = mask == torch.tensor(valid_obj_ids, dtype=torch.uint8)[:, None, None]
                labels = torch.ones((len(valid_obj_ids),), dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "masks": masks}
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(data_dirs, batch_size):
    dataloaders = {}
    for split, (image_dir, mask_dir) in data_dirs.items():
        dataset = MaskRCNNDataset(image_dir, mask_dir)
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            collate_fn=collate_fn,
            pin_memory=True,  # pinned memory
            num_workers=2,
            persistent_workers=True,
        )
    return dataloaders


# TorchScript
def get_model(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = torch.jit.script(model)
    return model.to(device)


# Train
def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training completed - Average Loss: {avg_loss:.4f}")
    return avg_loss


# IoU
def calculate_iou(pred, target, num_classes):
    ious = []
    pred = torch.argmax(pred, dim=1).to(device)
    target = target.to(device)
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
    pred = torch.argmax(pred, dim=1).to(device)
    target = target.to(device)
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


# Validate with metrics
def validate_model(model, val_loader, device, num_classes):
    model.eval()
    all_ious, all_dices = [], []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_masks = output['masks'].squeeze(1)
                true_masks = target['masks']

                # IoU
                ious = calculate_iou(pred_masks, true_masks, num_classes)

                # Dice
                dices = calculate_dice(pred_masks, true_masks, num_classes)

                all_ious.extend(ious)
                all_dices.extend(dices)

    # mean IoU, Dice
    mean_iou = torch.tensor(all_ious).nanmean().item()
    mean_dice = torch.tensor(all_dices).nanmean().item()

    return mean_iou, mean_dice

if __name__ == '__main__':

    dataloaders = get_dataloaders(DATA_DIRS, batch_size=batch_size)
    train_loader = dataloaders['train']
    val_loader = dataloaders['validation']

    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights, num_classes=91)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torch.nn.Sequential(
        torch.nn.Conv2d(in_features_mask, hidden_layer, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(hidden_layer, num_classes, kernel_size=1)
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    # Train and validate
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss = train_model(model, train_loader, optimizer, device)
        #mean_iou, mean_dice = validate_model(model, val_loader, device, num_classes)
        print(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")


'''
dataloaders = get_dataloaders(DATA_DIRS, batch_size)
train_loader = dataloaders['train']
val_loader = dataloaders['validation']

model = get_model(num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")

    # Train phase
    train_model(model, train_loader, optimizer)

    # Validation phase
    mean_iou, mean_dice = validate_model(model, val_loader, device, num_classes)
    print(f"Epoch {epoch + 1}/{epochs} - Mean IoU: {mean_iou:.4f}, Mean Dice: {mean_dice:.4f}")
'''
