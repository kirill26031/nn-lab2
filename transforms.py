import torch
import torchvision.transforms.v2 as T

def image_to_tensor(resize_size: int):
    return T.Compose([
    T.ToDtype(dtype=torch.float32, scale=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.Resize(size=(resize_size, resize_size)),
    ])

mask_to_tensor = T.Compose([
    T.ToImage(),
    T.ToDtype(dtype=torch.float32, scale=False)
])

def mask_of_uniform_size(resize_size: int):
    return T.Compose([
    T.Resize((resize_size, resize_size), interpolation=T.InterpolationMode.NEAREST_EXACT),
    mask_to_tensor,
])