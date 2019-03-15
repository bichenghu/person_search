# -----------------------------------------------------
# Spatial Transformer RPN of Person Search Architecture
#
# Created by: Liangqi Li
# Created Date: Apr 2, 2018
# -----------------------------------------------------
# -----------------------------------------------------
# Latest Modified By: Bicheng Hu
# Latest Modified Date: Mar 14, 2019
# -----------------------------------------------------
import yaml
import numpy.random as npr
import torch.nn as nn

import torch.nn.functional as func

from .anchor_compose import anchor_compose
from .anchor_target_layer import AnchorTargetLayer
from .proposal_layer import ProposalLayer
from .proposal_target_layer import _ProposalTargetLayer

from utils.bbox_transform import *
from utils.losses import smooth_l1_loss
from model.roi_layer import ROIAlign, ROIPool


def spatial_transform(bottom, trans_param):
    theta = trans_param.view(-1, 2, 3)
    # TODO: use different pooling size
    grid = func.affine_grid(theta, bottom.size())
    transformed = func.grid_sample(bottom, grid)

    return transformed


class STRPN(nn.Module):

    def __init__(self, net_conv_channels, num_pid):
        """
        create Spatial Transformer Region Proposal Network
        ---
        param:
            net_conv_channels: (int) channels of feature maps extracted by head
            training: (bool) training mode or test mode
        """
        super().__init__()
        with open('config.yml', 'r') as f:
            self.config = yaml.load(f)

        self.num_pid = num_pid
        self.feat_stride = self.config['rpn_feat_stride']
        self.rpn_channels = self.config['rpn_channels']
        self.anchor_scales = self.config['anchor_scales']
        self.anchor_ratios = self.config['anchor_ratios']
        self.pooling_size = self.config['pooling_size']
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)
        self.anchors = None  # to be set in other methods

        self.rpn_net = nn.Conv2d(
            net_conv_channels, self.rpn_channels, 3, padding=1)
        self.rpn_cls_score_net = nn.Conv2d(
            self.rpn_channels, self.num_anchors * 2, 1)
        self.rpn_bbox_pred_net = nn.Conv2d(
            self.rpn_channels, self.num_anchors * 4, 1)
        self.rpn_transform_net = nn.Conv2d(
            self.rpn_channels, self.num_anchors * 6, 1)

        self.anchor_target_layer = AnchorTargetLayer()
        self.proposal_layer = ProposalLayer(self.config)
        self.proposal_target_layer = _ProposalTargetLayer(self.config)

        self.roi_pooling = ROIPool((self.pooling_size, self.pooling_size), 1.0/16.0)
        self.roi_align = ROIAlign((self.pooling_size, self.pooling_size), 1.0/16.0, 0)

        self.initialize_weight(False)



    def forward(self, head_features, gt_boxes, im_info, mode='gallery'):
        if self.training:
            if mode == 'gallery':
                rois, rpn_info, label, bbox_info, roi_trans_param = \
                    self.region_proposal(head_features, gt_boxes, im_info)

                rpn_label, rpn_bbox_info, rpn_cls_score, rpn_bbox_pred = \
                    rpn_info

                rpn_cls_score = rpn_cls_score.view(-1, 2)
                rpn_label = rpn_label.view(-1)
                rpn_select = (rpn_label.data != -1).nonzero().view(-1)
                rpn_cls_score = rpn_cls_score.index_select(
                    0, rpn_select).contiguous().view(-1, 2)
                rpn_label = rpn_label.index_select(
                    0, rpn_select).contiguous().view(-1)

                rpn_cls_loss = func.cross_entropy(rpn_cls_score, rpn_label)
                rpn_box_loss = smooth_l1_loss(rpn_bbox_pred, rpn_bbox_info,
                                              sigma=3.0, dim=[1, 2, 3])
                rpn_loss = (rpn_cls_loss, rpn_box_loss)


                pooled_feat = self.roi_pooling(head_features, rois.view(-1, 5))

                transformed_feat = spatial_transform(
                    pooled_feat, roi_trans_param)

                return pooled_feat, transformed_feat, rpn_loss, label,\
                    bbox_info

            elif mode == 'query':
                pooled_feat = self.roi_pooling(head_features, gt_boxes, False)
                return pooled_feat

            else:
                raise KeyError(mode)

        else:
            if mode == 'gallery':
                rois, roi_trans_param = self.region_proposal(
                    head_features, gt_boxes, im_info)

                #pooled_feat = self.roi_pooling(head_features, rois.view(-1, 5))
                pooled_feat = self.pooling(head_features, rois)
                transformed_feat = spatial_transform(
                    pooled_feat, roi_trans_param)
                return rois, pooled_feat, transformed_feat

            elif mode == 'query':
                # TODO: whether to transform query
                pooled_feat = self.roi_pooling(head_features, gt_boxes.view(-1, 5))

                return pooled_feat

            else:
                raise KeyError(mode)

    def pooling(self, bottom, rois, max_pool=True):
        rois = rois.detach()
        x1 = (rois[:, 1::4] / 16.0).squeeze(1)
        y1 = (rois[:, 2::4] / 16.0).squeeze(1)
        x2 = (rois[:, 3::4] / 16.0).squeeze(1)
        y2 = (rois[:, 4::4] / 16.0).squeeze(1)

        height = bottom.size(2)
        width = bottom.size(3)

        # affine theta
        theta = rois.data.new(rois.size(0), 2, 3).zero_()
        theta[:, 0, 0] = (x2 - x1) / (width - 1)
        theta[:, 0, 2] = (x1 + x2 - width + 1) / (width - 1)
        theta[:, 1, 1] = (y2 - y1) / (height - 1)
        theta[:, 1, 2] = (y1 + y2 - height + 1) / (height - 1)

        pooling_size = self.config['pooling_size']
        if max_pool:
            pre_pool_size = pooling_size * 2
            grid = func.affine_grid(theta, torch.Size(
                (rois.size(0), 1, pre_pool_size, pre_pool_size)))
            crops = func.grid_sample(
                bottom.expand(rois.size(0), bottom.size(1), bottom.size(2),
                              bottom.size(3)), grid)
            crops = func.max_pool2d(crops, 2, 2)
        else:
            grid = func.affine_grid(theta, torch.Size(
                (rois.size(0), 1, pooling_size, pooling_size)))
            crops = func.grid_sample(
                bottom.expand(rois.size(0), bottom.size(1), bottom.size(2),
                              bottom.size(3)), grid)

        return crops



    def region_proposal(self, net_conv, gt_boxes, im_info):

        anchors = anchor_compose(self.config, net_conv.size(2), net_conv.size(3))

        rpn = func.relu(self.rpn_net(net_conv))
        rpn_cls_score = self.rpn_cls_score_net(rpn)
        rpn_cls_score_reshape = rpn_cls_score.view(
            1, 2, -1, rpn_cls_score.size()[-1])
        rpn_cls_prob_reshape = func.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = rpn_cls_prob_reshape.view_as(rpn_cls_score).permute(
            0, 2, 3, 1)
        rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1)
        rpn_cls_score_reshape = rpn_cls_score_reshape.permute(
            0, 2, 3, 1).contiguous()

        rpn_bbox_pred = self.rpn_bbox_pred_net(rpn)
        rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()

        rpn_trans_param = self.rpn_transform_net(rpn)
        rpn_trans_param = rpn_trans_param.permute(
            0, 2, 3, 1).contiguous()

        if self.training:
            rois, roi_scores, roi_trans_param = self.proposal_layer(
                anchors, rpn_cls_prob, rpn_bbox_pred, rpn_trans_param, im_info)
            rpn_labels, rpn_bbox_info = self.anchor_target_layer(
                rpn_cls_score, gt_boxes, im_info)
            rois, label, roi_trans_param, bbox_info = \
                self.proposal_target_layer(rois, roi_scores, roi_trans_param,
                                           gt_boxes)

            rpn_info = (rpn_labels, rpn_bbox_info, rpn_cls_score_reshape,
                        rpn_bbox_pred)

            return rois, rpn_info, label, bbox_info, roi_trans_param

        # test
        else:
            rois, _, roi_trans_param = self.proposal_layer(
                anchors, rpn_cls_prob, rpn_bbox_pred, rpn_trans_param, im_info)
            return rois, roi_trans_param





    def initialize_weight(self, trun):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initializer: truncated normal and random normal.
            """
            if truncated:
                # not a perfect approximation
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
            else:
                m.weight.data.normal_(mean, stddev)
            m.bias.data.zero_()

        normal_init(self.rpn_net, 0, 0.01, trun)
        normal_init(self.rpn_cls_score_net, 0, 0.01, trun)
        normal_init(self.rpn_bbox_pred_net, 0, 0.01, trun)
        # TODO: change bias for rpn_transform_net
        normal_init(self.rpn_transform_net, 0, 0.01, trun)


