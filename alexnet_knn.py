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


def retrieve_features(input, layer_name="classifier.0", sliding_window_size=32, 
                         sliding_window_step=11, feature_depth=9216, model = get_model(device)):
    image_to_sliding_patches = T.Compose([
        SlidingWindow(sliding_window_size, sliding_window_step)
    ])
    upscale = T.Compose([T.Resize(size=(224, 224))])
    patches = image_to_sliding_patches(input)
    features = torch.zeros(patches.size(0), patches.size(1), feature_depth, device=device)

    patches_dataset = data_utils.TensorDataset(patches)
    patches_loader = DataLoader(patches_dataset, batch_size=1, shuffle=False)

    i_h = 0
    for batch in tqdm.tqdm(patches_loader):
        upscaled = upscale(batch[0][0])
        model(upscaled)
        feature = get_feature(upscaled, get_extractor(device, model, layer_name), layer_name)
        # print(i_h, ", ", feature.shape)
        features[i_h] = feature.view(batch[0][0].size(0), -1)
        i_h += 1
    return features


def perform_segmentation(input, ground_truth, layer_name="classifier.0", sliding_window_size=32, 
                         sliding_window_step=11, feature_depth=9216, model = get_model(device)):
    features = retrieve_features(input, layer_name, sliding_window_size, 
                         sliding_window_step, feature_depth, model)
    mask_patches = SlidingWindow(sliding_window_size, sliding_window_step)(ground_truth)
    print(mask_patches.shape)
    
    patches_height = features.size(0)
    patches_width = features.size(1)
    torch.cuda.empty_cache()

    image_means = KMeans(device, features.view(-1, features.size(2)), 5, n_iters=10)
    torch.cuda.empty_cache()
    predicted_classes = image_means.predict(features.view(-1, features.size(2)))
    torch.cuda.empty_cache()
    predicted_classes = predicted_classes.view(patches_height, patches_width)
    del features
    torch.cuda.empty_cache()
    return predicted_classes

def segmentation_image(predicted_classes: torch.Tensor, size=(224, 224)):
    mask_to_image = T.Compose([
        T.ToTensor(),
        T.ToDtype(torch.float32),
        T.Normalize(mean=[0.0014], std=[0.0031]),
        T.ToPILImage(),
        T.Resize(size=size)
    ])
    return mask_to_image(predicted_classes.unsqueeze(0))


import pickle
import os
import lz4

def save_features(index, features, src_path, layer_name="classifier.0", sliding_window_size=32, 
                         sliding_window_step=11, name='train'):
    file_name = os.path.join(src_path, "features", 
                            "{name}_{layer_name}_{window_size}_{window_step}_{i}.lz4"
                              .format(name=name, layer_name=layer_name, window_size=sliding_window_size, 
                                      window_step=sliding_window_step, i=index)
                            )
    with lz4.frame.open(file_name, mode="wb") as f:
      pickle.dump(features.shape , f) 
      pickle.dump(features, f)

def read_features(index, src_path, layer_name="classifier.0", sliding_window_size=32, 
                         sliding_window_step=11, name="train"):
    file_name = os.path.join(src_path, "features", 
                            "{name}_{layer_name}_{window_size}_{window_step}_{i}.lz4"
                              .format(layer_name=layer_name, window_size=sliding_window_size, 
                                      window_step=sliding_window_step, i=index, name=name)
                            )
    with lz4.frame.open(file_name, mode="rb") as f:
      features_shape = pickle.load(f)
      features = pickle.load(f)
      return features
    
def load_and_merge_knn_features(device, src_path, reducer=None, indices = range(5), name="train"):
    knn_features = []
    for i in indices:
        feature = read_features(i, src_path, name=name).to("cpu")
        feature = feature.view(feature.size(0) * feature.size(1), -1)
        if reducer is None:
            knn_features.append(feature)
        else:
            knn_features.append(torch.tensor(reducer.transform(feature)))
        torch.cuda.empty_cache()

    return torch.cat(knn_features, dim=0).to(device)

