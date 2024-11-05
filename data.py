from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.colors import ListedColormap
from PIL import Image, ExifTags

dataset = load_dataset("farmaieu/plantorgans")

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
