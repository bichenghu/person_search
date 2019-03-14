# -----------------------------------------------------
# Anchor Target Layer
#
# Created By: Bicheng Hu
# Created Date: 2019/3/14
# -----------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from utils.generate_anchors import generate_anchors
from utils.bbox_transform import bbox_transform, bbox_overlaps

class AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __int__(self, feat_stride, scales, ratios):
        super(AnchorTargetLayer, self).__init__()

        self.feat_stride = feat_stride
        self.anchor_scales = scales
        self.anchor_ratios = ratios
















