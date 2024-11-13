import os
from PIL import Image
from datasets import load_dataset, DatasetDict
from data import DATA_PATH

dataset = load_dataset("farmaieu/plantorgans", cache_dir=DATA_PATH)

# розподіл на тренувальні та тестові дані (80/20)
dataset_split = dataset['train'].train_test_split(test_size=0.2, seed=42)

# валідаційні дані (10% від тренувальних)
train_val_split = dataset_split['train'].train_test_split(test_size=0.1, seed=42)

dataset_final = DatasetDict({
    'train': train_val_split['train'],
    'validation': train_val_split['test'],
    'test': dataset_split['test']
})

output_dirs = ['train/images', 'train/masks', 'validation/images', 'validation/masks', 'test/images', 'test/masks']
for directory in output_dirs:
    os.makedirs(os.path.join(DATA_PATH, directory.replace('/', '\\')), exist_ok=True)


def save_image_and_mask(image, mask, image_path, mask_path):
    image.save(image_path)
    mask.save(mask_path)


def save_dataset_to_files(dataset_split, split_name):
    for idx, example in enumerate(dataset_split):
        image_filename = f"{split_name}/images/img_{idx}.png"
        mask_filename = f"{split_name}/masks/mask_{idx}.png"

        image = example['image']
        mask = example['label']

        save_image_and_mask(image, mask, 
                            os.path.join(DATA_PATH, image_filename.replace('/', '\\')), 
                            os.path.join(DATA_PATH, mask_filename.replace('/', '\\')))


save_dataset_to_files(dataset_final['train'], 'train')
save_dataset_to_files(dataset_final['validation'], 'validation')
save_dataset_to_files(dataset_final['test'], 'test')