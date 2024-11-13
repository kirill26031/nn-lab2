import numpy as np

from data import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    show_mask()
    get_dataset_info()
    class_distribution = get_class_distribution()
    get_average_sizes(class_distribution)
    '''
    class_distribution = {
        np.uint8(0): np.int64(67720594331), 
        np.uint8(1): np.int64(4390472503), 
        np.uint8(2): np.int64(7192139908), 
        np.uint8(4): np.int64(1775289245), 
        np.uint8(3): np.int64(5756712)
        }
    #plot_class_frequency(class_distribution)
    example_mask = np.array(dataset['train'][0]['label'])

    unique_classes = np.unique(example_mask)

    plt.figure(figsize=(15, 5))

    for i, class_idx in enumerate(unique_classes):
        highlighted = show_class(example_mask, class_idx)

        plt.subplot(1, len(unique_classes), i + 1)
        plt.imshow(highlighted, cmap=ListedColormap(['black', 'red', 'green', 'blue', 'yellow', 'purple']))
        plt.title(f'Клас {class_idx}')
        plt.axis('off')

    plt.show()

