import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vgg16, VGG16_Weights
from torchvision import models


class FCN (nn.Module):
    def __init__(self, num_classes, backbone='resnet', num_skip_connections=3, dropout_rate=0.3):
        super(FCN, self).__init__()

        if backbone == 'resnet':
            base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.encoder_layers = nn.Sequential(
                base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool,
                base_model.layer1, base_model.layer2, base_model.layer3, base_model.layer4
            )
            encoder_channels = [64, 256, 512, 1024, 2048]
            if backbone == 'vgg':
                base_model = models.vgg16(weights=VGG16_Weights.DEFAULT)
                self.encoder_layers = base_model.features
                encoder_channels = [64, 128, 256, 512, 512]
        else:
            raise ValueError("Backbone must be 'resnet' or 'vgg'.")

        self.num_skip_connections = min(num_skip_connections, len(encoder_channels))

        # skip connections
        self.skip_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_channels[-(i + 1)], num_classes, kernel_size=1),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True)
            )
            for i in range(self.num_skip_connections)
        ])

        # upsampling
        self.upscore = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True)
            )
            for _ in range(self.num_skip_connections)
        ])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        input_size = x.size()
        features = []
        for layer in self.encoder_layers:
            x = layer(x)
            features.append(x)

        features = features[-self.num_skip_connections:]

        output = self.skip_convs[0](features[-1])

        # Upsampling with skip connections
        for i in range(1, self.num_skip_connections):
            output = self.upscore[i - 1](output)
            output = output + self.skip_convs[i](features[-(i + 1)])

        output = F.interpolate(output, size=(input_size[2], input_size[3]), mode="bilinear", align_corners=False)

        output = self.dropout(output)
        return output