import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms.v2 import Transform
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

def image_to_patches(image: torch.Tensor, patch_size=32):
    if image.dim() == 3:
        image = image.unsqueeze(0)

    _, _, H, W = image.shape
    pad_height = (patch_size - H % patch_size) % patch_size
    pad_width = (patch_size - W % patch_size) % patch_size
    
    # Pad the image if necessary
    padding = (0, pad_width, 0, pad_height)
    image_padded = F.pad(image, padding, mode='constant', value=0)

    # Use unfold to extract patches
    patches = image_padded.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # Reshape to get a tensor of shape (N, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(-1, image_padded.size(1), patch_size, patch_size)
    
    return patches

def sliding_window_patches(image: torch.Tensor, kernel_size, stride=1):
    _, _, H, W = image.shape
    first_pad = kernel_size // 2
    second_pad = first_pad
    if kernel_size % 2 == 0:
        second_pad = first_pad - 1

    padding = (first_pad, second_pad, first_pad, second_pad)
    img_padded = torch.nn.functional.pad(image, padding, mode='replicate')
    # Use unfold to extract patches
    patches = img_padded.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)

    # Reshape to get a tensor of shape (N, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(patches.size(1), patches.size(2), img_padded.size(1), kernel_size, kernel_size)
    return patches

def patch_index_to_position(image_width: int, image_height: int, index: int, patch_size=32):
    # Calculate the number of patches along the height and width
    patches_per_row = (image_width - patch_size) // patch_size + 1
    patches_per_col = (image_height - patch_size) // patch_size + 1

    # Calculate the row and column position of the patch
    row = (index // patches_per_row)
    col = (index % patches_per_col)

    return (row, col)

class MyTransform(Transform):
    def __init__(self, patch_size=32):
        super().__init__()
        self.patch_size = patch_size

    def __call__(self, input, **other):
        patches = image_to_patches(input, self.patch_size)

        if other is None or len(other) == 0:
            return patches
        
        other_pathes = image_to_patches(other[0], self.patch_size)
        return patches, other_pathes
    
class SlidingWindow(Transform):
    def __init__(self, patch_size=32, stride=1):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, input, **other):
        patches = sliding_window_patches(input, self.patch_size, self.stride)

        if other is None or len(other) == 0:
            return patches
        
        other_pathes = sliding_window_patches(other[0], self.patch_size, self.stride)
        return patches, other_pathes
    
class ExtractFeatures(Transform):
    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor

    def __call__(self, input, **other):
        features = get_feature(input, self.extractor)

        if other is None or len(other) == 0:
            return features
        
        other_features = get_feature(other[0], self.extractor)
        return features, other_features
    
def get_extractor(device, model, layer_name):
    return_nodes = {
        layer_name: layer_name
    }
    extractor = create_feature_extractor(model, return_nodes=return_nodes).to(device)
    extractor.eval()
    extractor.requires_grad_(False)
    return extractor

def get_feature(input, model, layer_name):
    return model(input)[layer_name]

def get_model(device):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True).to(device)
    model.eval()
    model.requires_grad_(False)
    return model