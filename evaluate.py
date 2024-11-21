import torch
import torchmetrics
from torchmetrics.segmentation import MeanIoU
from preprocessing import preprocess_image_and_mask
import torchvision.transforms.v2 as T
from alexnet import get_model, SlidingWindow, ExtractFeatures, get_extractor, get_feature
from train import device
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from kmeans import KMeans
import tqdm

def calculate_metrics(device, ground_truth_mask, predicted_class):
    mask_to_tensor = T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=False),
    ])
    predicted_mask = predicted_class.unsqueeze(0).to(device)
    ground_truth_transform = T.Compose([
        T.Resize(size=(predicted_mask.size(1), predicted_mask.size(2)), interpolation=T.InterpolationMode.NEAREST_EXACT),
        T.ToDtype(dtype=torch.int64)
    ])
    ground_truth = ground_truth_transform(mask_to_tensor(ground_truth_mask)).to(device=device)

    acc = torchmetrics.functional.accuracy(predicted_mask, ground_truth, task="multiclass", num_classes=5)
    meanIoU = MeanIoU(num_classes=5, input_format='index', include_background=True, per_class=True).to(device)
    meanIoU_result = meanIoU(predicted_mask, ground_truth)
    return {
        "accuracy": acc.item(),
        "meanIoU": meanIoU_result.item()
    }

