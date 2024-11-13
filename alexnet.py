import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.transforms.v2 import Transform

class AlexNet(nn.Module):
    def __init__(self, kernel_sizes, conv_sizes, paddings, fc_sizes, stride0=4, image_size=224):
        super().__init__()
        self.image_size = image_size
        self.kernel_sizes = kernel_sizes
        self.conv_sizes = conv_sizes
        self.paddings = paddings
        self.fc_sizes = fc_sizes
        self.stride0 = stride0

        self.conv1 = nn.Conv2d(3, conv_sizes[0], kernel_sizes[0], stride0, padding=paddings[0])
        self.conv2 = nn.Conv2d(conv_sizes[0], conv_sizes[1], kernel_sizes[1], padding=paddings[1])
        self.conv3 = nn.Conv2d(conv_sizes[1], conv_sizes[2], kernel_sizes[2], padding=paddings[2])
        self.conv4 = nn.Conv2d(conv_sizes[2], conv_sizes[3], kernel_sizes[3], padding=paddings[3])
        self.conv5 = nn.Conv2d(conv_sizes[3], conv_sizes[4], kernel_sizes[4], padding=paddings[4])

        self.after_conv_size = self.calculate_after_conv_size()

        self.fc1 = nn.Linear(conv_sizes[4] * self.after_conv_size * self.after_conv_size, fc_sizes[0])
        self.fc2 = nn.Linear(fc_sizes[0], fc_sizes[1])
        self.fc3 = nn.Linear(fc_sizes[1], 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.conv_sizes[4] * self.after_conv_size * self.after_conv_size)
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def calculate_after_conv_size(self):
        after_conv1 = int((self.image_size - self.kernel_sizes[0] + self.paddings[0] + self.stride0 - 2) // self.stride0) // 2
        after_conv2 = (after_conv1 - self.kernel_sizes[1] + self.paddings[1] + 1 - 2) // 2
        after_conv4 = after_conv2 - self.kernel_sizes[2] - self.kernel_sizes[3] + self.paddings[2] + self.paddings[3] + 2
        after_conv5 = (after_conv4 - self.kernel_sizes[4] + self.paddings[4] + 1 - 2) // 2
        return int(after_conv5)

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
        # print(other)
        other_pathes = image_to_patches(other[0], self.patch_size)
        return patches, other_pathes
