import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
from data import DATA_PATH


def resize_image_and_mask(image, mask, size=(256, 256)):
    resized_image = image.resize(size, Image.BILINEAR)
    resized_mask = mask.resize(size, Image.NEAREST)
    return resized_image, resized_mask


def normalize_image(image):
    image = np.array(image).astype(np.float32) / 255.0
    return image


def augment_image_and_mask(image, mask):
    augment = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
    ])
    return augment(image), augment(mask)


def random_crop(image, mask, crop_size=(128, 128)):
    i, j, h, w = T.RandomCrop.get_params(image, output_size=crop_size)
    cropped_image = T.functional.crop(image, i, j, h, w)
    cropped_mask = T.functional.crop(mask, i, j, h, w)
    return cropped_image, cropped_mask


def mask_to_class_array(mask):
    return np.array(mask).astype(np.uint8)


DATA_DIRS = {
    'train': ('train/images', 'train/masks'),
    'validation': ('validation/images', 'validation/masks'),
    'test': ('test/images', 'test/masks')
}


def preprocess_image_and_mask(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  # Маска у відтінках сірого

    image, mask = resize_image_and_mask(image, mask)

    # augmenting train only
    if 'train' in image_path:
        image, mask = augment_image_and_mask(image, mask)

    image = normalize_image(image)

    mask = mask_to_class_array(mask)

    return image, mask


def preprocess_dataset(data_directory):
    image_dir, mask_dir = data_directory

    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))

    processed_images = []
    processed_masks = []

    for img_file, mask_file in zip(image_files, mask_files):
        image_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        image, mask = preprocess_image_and_mask(image_path, mask_path)

        processed_images.append(image)
        processed_masks.append(mask)

    return processed_images, processed_masks


# processed_data = {}
# for split_name, data_directory in DATA_DIRS.items():
#     print(f"Processing {split_name}...")
#     images, masks = preprocess_dataset([
#         os.path.join(DATA_PATH, data_directory[0].replace('/', '\\')),
#         os.path.join(DATA_PATH, data_directory[1].replace('/', '\\'))
#         ])
#     processed_data[split_name] = {'images': images, 'masks': masks}

# print("Preprocessing finished")