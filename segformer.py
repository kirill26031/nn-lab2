from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import shap

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 8
learning_rate = 0.0001
dropout_rate = 0.3
weight_decay = 0.0001
epochs = 5
target_size = (256, 256)


# Dataset Definition
class PlantOrgansDataset(Dataset):
    def __init__(self, image_dir, mask_dir, feature_extractor, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.feature_extractor = feature_extractor
        self.target_size = target_size

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        image = np.array(image)
        mask = np.array(mask)

        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(), torch.tensor(mask, dtype=torch.long)


model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=5)


def get_dataloaders(image_dir, mask_dir, feature_extractor, batch_size, target_size=(256, 256)):
    dataset = PlantOrgansDataset(image_dir, mask_dir, feature_extractor, target_size=target_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


train_loader = get_dataloaders('train/images', 'train/masks', feature_extractor, batch_size=8)
val_loader = get_dataloaders('validation/images', 'validation/masks', feature_extractor, batch_size=8)


def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(pixel_values=images, labels=masks)
            val_loss += criterion(outputs.logits, masks).item()
    val_loss /= len(val_loader)
    return val_loss


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=5, device="cuda"):
    model.to(device)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=images, labels=masks)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = validate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


# Training
print("="*40)
print("Starting Training")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {learning_rate}")
print(f"Dropout Rate: {dropout_rate}")
print(f"Weight Decay: {weight_decay}")
print(f"Target Image Size: {target_size}")
print(f"Number of Epochs: {epochs}")
print("="*40)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()
train_loader = get_dataloaders('train/images', 'train/masks', feature_extractor, batch_size, target_size=target_size)
val_loader = get_dataloaders('validation/images', 'validation/masks', feature_extractor, batch_size, target_size=target_size)
train_model(model, train_loader, val_loader, optimizer, criterion, epochs=epochs, device=device)


# Activation Maps (Attention Maps)
def show_attention_maps(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=image.unsqueeze(0).to(device))
    logits = outputs.logits.squeeze(0)
    attention_maps = logits.argmax(dim=0).cpu().numpy()
    plt.imshow(attention_maps, cmap="jet")
    plt.title("Attention Map")
    plt.show()


# Show attention maps for a sample
sample_image, _ = next(iter(val_loader))
show_attention_maps(model, sample_image[0])


# SHAP for SegFormer
def explain_with_shap(model, dataloader):
    def model_prediction(inputs):
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(pixel_values=inputs)
        return outputs.logits.cpu().numpy()

    sample_batch = next(iter(dataloader))[0].numpy()
    explainer = shap.DeepExplainer(model_prediction, sample_batch[:8])
    shap_values = explainer.shap_values(sample_batch[:8])
    shap.image_plot(shap_values, sample_batch[:8])


# Explain predictions with SHAP
explain_with_shap(model, val_loader)
