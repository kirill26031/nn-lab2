import torch
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


def create_mask_rcnn(num_classes, anchor_sizes, num_proposals):

    # Backbone: ResNet-50 + FPN
    backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=None)

    anchor_generator = AnchorGenerator(
        sizes=(anchor_sizes,),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # ROI Align
    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )
    mask_roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=None,
        box_roi_pool=roi_pooler,
        mask_roi_pool=mask_roi_pooler,
        rpn_pre_nms_top_n_train=num_proposals,
        rpn_post_nms_top_n_train=num_proposals
    )
    return model
