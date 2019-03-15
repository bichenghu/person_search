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
import yaml

from utils.generate_anchors import generate_anchors
from utils.bbox_transform import bbox_transform, bbox_overlaps

class AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __int__(self):
        super(AnchorTargetLayer, self).__init__()
        with open('config.yml', 'r') as f:
           self.config = yaml.load(f)

    def forward(self, rpn_cls_score, gt_boxes, im_info):
        def _unmap(data, count, inds, fill=0):
            """
            Unmap a subset of item (data) back to the original set of items
            (of size count)
            """
            if len(data.shape) == 1:
                ret = np.empty((count,), dtype=np.float32)
                ret.fill(fill)
                ret[inds] = data
            else:
                ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
                ret.fill(fill)
                ret[inds, :] = data
            return ret

        def _compute_targets(ex_rois, gt_rois):
            """Compute bounding-box regression targets for an image."""

            assert ex_rois.shape[0] == gt_rois.shape[0]
            assert ex_rois.shape[1] == 4
            assert gt_rois.shape[1] >= 5

            # add float convert
            return bbox_transform(torch.from_numpy(ex_rois),
                                  torch.from_numpy(gt_rois[:, :4])).numpy()

        all_anchors = self.anchors.data.cpu().numpy()
        gt_boxes = gt_boxes.data.cpu().numpy()
        rpn_cls_score = rpn_cls_score.data

        num_anchor = self.num_anchors
        total_anchors = all_anchors.shape[0]

        # allow boxes to sit over the edge by a small amount
        _allowed_border = 0

        # map of shape (..., H, W)
        height, width = rpn_cls_score.shape[1:3]

        # only keep anchors inside the image
        inds_inside = np.where(
            (all_anchors[:, 0] >= -_allowed_border) &
            (all_anchors[:, 1] >= -_allowed_border) &
            (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
            (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
        )[0]

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt)
        overlaps = bbox_overlaps(
            np.ascontiguousarray(anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        if not self.config['train_rpn_clobber_positive']:
            # assign bg labels first so that positive labels can clobber them
            # first set the negatives
            labels[max_overlaps < self.config['train_rpn_neg_overlap']] = 0

            # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= self.config['train_rpn_pos_overlap']] = 1

        if self.config['train_rpn_clobber_positive']:
            # assign bg labels last so that negative labels can clobber pos
            labels[max_overlaps < self.config['train_rpn_neg_overlap']] = 0

        # subsample positive labels if we have too many
        num_fg = int(self.config['train_rpn_fg_frac'] *
                     self.config['train_rpn_batchsize'])
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = self.config['train_rpn_batchsize'] - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        # only the positive ones have regression targets
        bbox_inside_weights[labels == 1, :] = np.array(
            self.config['train_rpn_bbox_inside_weights'])

        bbox_outside_weights = np.zeros((len(inds_inside), 4),
                                        dtype=np.float32)
        if self.config['train_rpn_pos_weight'] < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((self.config['train_rpn_pos_weight'] > 0) &
                    (self.config['train_rpn_pos_weight'] < 1))
            positive_weights = (self.config['train_rpn_pos_weight'] /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - self.config['train_rpn_pos_weight']) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors,
                                     inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors,
                                      inds_inside, fill=0)

        # labels
        labels = labels.reshape(
            (1, height, width, num_anchor)).transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, num_anchor * height, width))
        rpn_labels = torch.from_numpy(labels).float().cuda().long()

        # bbox_targets
        bbox_targets = bbox_targets.reshape((1, height, width, num_anchor * 4))

        rpn_bbox_targets = torch.from_numpy(bbox_targets).float().cuda()
        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights.reshape(
            (1, height, width, num_anchor * 4))
        rpn_bbox_inside_weights = torch.from_numpy(
            bbox_inside_weights).float().cuda()

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights.reshape(
            (1, height, width, num_anchor * 4))
        rpn_bbox_outside_weights = torch.from_numpy(
            bbox_outside_weights).float().cuda()

        return rpn_labels, (rpn_bbox_targets, rpn_bbox_inside_weights,
                            rpn_bbox_outside_weights)