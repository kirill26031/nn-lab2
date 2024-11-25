from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from PIL import Image, ExifTags
from torch.utils.data.dataset import Dataset


DATA_PATH = "C:\\Users\\pc\\Documents\\repos\\mp-2\\nn\\nn-lab2\\data"
dataset = load_dataset("farmaieu/plantorgans", cache_dir=DATA_PATH, verification_mode="no_checks")

image = dataset['train'][0]['image']
mask = dataset['train'][0]['label']
#labels = dataset['train'].features['label'].names

def show_mask():
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Зображення рослини')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Маска сегментації')

    plt.show()


def get_dataset_info():
    print("Розмір датасету:", len(dataset['train']))
    print("Формат даних:", dataset['train'].features)


def get_class_distribution():
    class_distribution = {}

    for example in dataset['train']:
        mask = np.array(example['label'])
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            if u in class_distribution:
                class_distribution[u] += c
            else:
                class_distribution[u] = c

    print("Розподіл класів:", class_distribution)
    return class_distribution


def get_average_sizes(class_distribution):
    object_sizes = {label: [] for label in class_distribution.keys()}

    for example in dataset['train']:
        mask = np.array(example['label'])
        for label_idx in class_distribution.keys():
            object_sizes[label_idx].append(np.sum(mask == label_idx))

    for label, sizes in object_sizes.items():
        print(f"Середній розмір для класу {label}: {np.mean(sizes):.2f} пікселів")


def plot_class_frequency (class_distribution):
    plt.figure(figsize=(10, 5))
    plt.bar(class_distribution.keys(), class_distribution.values(), color='skyblue')
    plt.title('Частка пікселів для кожного класу')
    plt.xlabel('Клас')
    plt.ylabel('Кількість пікселів (логарифмічна шкала)')
    plt.yscale('log')  # Застосовуємо логарифмічну шкалу до осі Y, бо у класу 3 мало пікселів
    plt.xticks(list(class_distribution.keys()))
    plt.show()


def show_class(mask, class_index):
    highlighted_image = np.zeros_like(mask)
    highlighted_image[mask == class_index] = class_index

    return highlighted_image

class PlantOrgansDataset(Dataset):
    def __init__(self, dataset, common_transform=None, images_transform=None, masks_transform=None):
        # self.X = dataset[tag]['image']
        # self.y = dataset[tag]['label']
        self.dataset = dataset
        self.count = len(dataset)
        self.common_transform = common_transform
        self.images_transform = images_transform
        self.masks_transform = masks_transform

    def __getitem__(self, index):
        X = self.dataset[index]['image']
        y = self.dataset[index]['label']
        size = self.dataset[index]['image'].size
        if self.common_transform is not None:
            X, y = self.common_transform(X, y)
            # size = (X.size(1), X.size(2))
            # print(X.shape)
        if self.images_transform is not None:
            X = self.images_transform(X)
        if self.masks_transform is not None:
            y = self.masks_transform(y)
        return X, y

    def __len__(self):
        return self.count
    
