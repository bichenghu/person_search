# -----------------------------------------------------
# Proposal Layer
#
# Created By: Bicheng Hu
# Created Date: 2019/3/14
# -----------------------------------------------------
import torch
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from ..nms import nms

