# -----------------------------------------------------
# Proposal Layer
#
# Created By: Bicheng Hu
# Created Date: 2019/3/14
# -----------------------------------------------------
import torch
import torch.nn as nn
from utils.bbox_transform import bbox_transform_inv, clip_boxes
from ..nms import nms

class ProposalLayer(nn.Module):
    def __init__(self, config):
        super(ProposalLayer, self).__init__()
        self.config = config
        self.anchor_scales = self.config['anchor_scales']
        self.anchor_ratios = self.config['anchor_ratios']
        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

    def forward(self, anchors, rpn_cls_prob, rpn_bbox_pred, rpn_trans_param,
                       im_info):
        if self.training:
            pre_nms_top_n = self.config['train_rpn_pre_nms_top_n']
            post_nms_top_n = self.config['train_rpn_post_nms_top_n']
            nms_thresh = self.config['train_rpn_nms_thresh']
        else:
            pre_nms_top_n = self.config['test_rpn_pre_nms_top_n']
            post_nms_top_n = self.config['test_rpn_post_nms_top_n']
            nms_thresh = self.config['test_rpn_nms_thresh']

        # Get the scores and bounding boxes
        scores = rpn_cls_prob[:, :, :, self.num_anchors:]
        rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))
        scores = scores.contiguous().view(-1, 1)
        rpn_trans_param = rpn_trans_param.view((-1, 6))

        proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
        proposals = clip_boxes(proposals, im_info[:2])

        # proposals = proposals.unsqueeze(0)

        # Pick the top region proposals
        # scores, order = scores.view(-1).sort(descending=True)
        scores, order = torch.sort(scores.view(-1), 0, True)
        if pre_nms_top_n > 0:
            order = order[:pre_nms_top_n]
            scores = scores[:pre_nms_top_n].view(-1)
        # proposals = proposals[order.data, :]
        proposals = proposals[order.data, :]
        trans_param = rpn_trans_param[order.data, :]

        # Non-maximal suppression
        # keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)
        keep = nms(proposals, scores, nms_thresh)

        # Pick th top region proposals after NMS
        if post_nms_top_n > 0:
            keep = keep[:post_nms_top_n]
        proposals = proposals[keep, :]
        scores = scores[keep,]
        trans_param = trans_param[keep, :]

        # Only support single image as input
        batch_inds = proposals.data.new(proposals.size(0), 1).zero_()
        blob = torch.cat((batch_inds, proposals), 1)

        return blob, scores, trans_param