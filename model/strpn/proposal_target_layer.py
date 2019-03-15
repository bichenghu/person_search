# -----------------------------------------------------
# Proposal Target Layer
#
# Created By: Bicheng Hu
# Created Date: 2019/3/14
# -----------------------------------------------------
import torch
import numpy as np
import numpy.random as npr

from utils.bbox_transform import bbox_transform, bbox_overlaps

class _ProposalTargetLayer():
    def __init__(self, config):
        super(_ProposalTargetLayer, self).__init__()
        self.config = config



    def forward(self, rpn_rois, rpn_scores, trans_param,
                              gt_boxes):
        """
        Assign object detection proposals to ground-truth targets. Produces
        proposal classification labels and bounding-box regression targets.
        """

        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source

        def _get_bbox_regression_labels(bbox_target_data, num_classes):
            """Bounding-box regression targets (bbox_target_data) are stored in
            a compact form N x (class, tx, ty, tw, th)

            This function expands those targets into the 4-of-4*K
            representation used by the network (i.e. only one class
            has non-zero targets).

            Returns:
                bbox_target (ndarray): N x 4K blob of regression targets
                bbox_inside_weights (ndarray): N x 4K blob of loss weights
            """
            # Inputs are tensor

            clss = bbox_target_data[:, 0]
            bbox_tar = clss.new(clss.numel(), 4 * num_classes).zero_()
            bbox_in_weights = clss.new(bbox_tar.shape).zero_()
            inds = (clss > 0).nonzero().view(-1)
            if inds.numel() > 0:
                clss = clss[inds].contiguous().view(-1, 1)
                dim1_inds = inds.unsqueeze(1).expand(inds.size(0), 4)
                dim2_inds = torch.cat(
                    [4 * clss, 4 * clss + 1, 4 * clss + 2, 4 * clss + 3],
                    1).long()
                bbox_tar[dim1_inds, dim2_inds] = bbox_target_data[inds][:, 1:]
                tr_bb_in_wei = self.config['train_bbox_inside_weights']
                bbox_in_weights[dim1_inds, dim2_inds] = bbox_tar.new(
                    tr_bb_in_wei).view(-1, 4).expand_as(dim1_inds)

            return bbox_tar, bbox_in_weights

        def _compute_targets(ex_rois, gt_rois, label):
            """Compute bounding-box regression targets for an image."""
            # Inputs are tensor

            assert ex_rois.shape[0] == gt_rois.shape[0]
            assert ex_rois.shape[1] == 4
            assert gt_rois.shape[1] == 4

            targets = bbox_transform(ex_rois, gt_rois)
            if self.config['train_bbox_normalize_targets_precomputed']:
                # Optionally normalize targets by a precomputed mean and stdev
                means = self.config['train_bbox_normalize_means']
                stds = self.config['train_bbox_normalize_stds']
                targets = ((targets - targets.new(means)) / targets.new(stds))
            return torch.cat([label.unsqueeze(1), targets], 1)

        def _sample_rois(al_rois, al_scores, tr_param, gt_box, fg_rois_per_im,
                         rois_per_im, num_classes, num_pid):
            """Generate a random sample of RoIs comprising foreground and
            background examples.
            """
            # overlaps: (rois x gt_boxes)
            overlaps = bbox_overlaps(
                al_rois[:, 1:5].data,
                gt_box[:, :4].data)
            max_overlaps, gt_assignment = overlaps.max(1)
            label = gt_box[gt_assignment, [4]]

            # Select foreground RoIs as those with >= FG_THRESH overlap
            fg_inds = (max_overlaps >=
                       self.config['train_fg_thresh']).nonzero().view(-1)
            # Guard against when an image has fewer than fg_rois_per_image

            # # ========================added=======================
            # # foreground RoIs
            # fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size(0))
            # # Sample foreground regions without replacement
            # if fg_inds.size(0) > 0:
            #   fg_inds = fg_inds[torch.from_numpy(
            #     npr.choice(np.arange(0, fg_inds.numel()), size=int(
            # fg_rois_per_this_image), replace=False)).long().cuda()]
            # # ====================================================

            # Select bg RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            bg_inds = ((max_overlaps < self.config['train_bg_thresh_hi']) +
                       (max_overlaps >= self.config['train_bg_thresh_lo'])
                       == 2).nonzero().view(-1)

            # =========================origin==========================
            # Small modification to the original version where we ensure a
            # fixed number of regions are sampled
            if fg_inds.numel() > 0 and bg_inds.numel() > 0:
                fg_rois_per_im = min(fg_rois_per_im, fg_inds.numel())

                if gt_box.size(0) < fg_rois_per_im:
                    gt_inds = torch.from_numpy(np.arange(
                        0, gt_box.size(0))).long().cuda()
                    fg_inds = torch.cat((gt_inds, fg_inds[torch.from_numpy(
                        npr.choice(np.arange(gt_box.size(0), fg_inds.numel()),
                                   size=int(fg_rois_per_im) - gt_box.size(0),
                                   replace=False)).long().cuda()]))
                else:
                    lab_inds = (gt_box[:, 5] != -1).nonzero().squeeze(-1)
                    if -1 in gt_box[:, 5].data:
                        unlab_inds = (gt_box[:, 5] == -1).nonzero().squeeze(-1)
                        fg_inds = torch.cat((lab_inds, torch.from_numpy(
                            npr.choice(unlab_inds.cpu().numpy(),
                                       size=fg_rois_per_im - lab_inds.numel(),
                                       replace=False)).long().cuda()))
                    else:
                        fg_inds = lab_inds

                # # ======================original========================
                # fg_inds = fg_inds[torch.from_numpy(
                #     npr.choice(np.arange(0, fg_inds.numel()),
                #                size=int(fg_rois_per_im),
                #                replace=False)).long().cuda()]
                # fg_inds = torch.from_numpy(
                #     (np.sort(fg_inds.cpu().numpy()))).long().cuda()

                bg_rois_per_im = rois_per_im - fg_rois_per_im
                to_replace = bg_inds.numel() < bg_rois_per_im
                bg_inds = bg_inds[torch.from_numpy(
                    npr.choice(np.arange(0, bg_inds.numel()),
                               size=int(bg_rois_per_im),
                               replace=to_replace)).long().cuda()]
            elif fg_inds.numel() > 0:
                to_replace = fg_inds.numel() < rois_per_im
                fg_inds = fg_inds[torch.from_numpy(
                    npr.choice(np.arange(0, fg_inds.numel()),
                               size=int(rois_per_im),
                               replace=to_replace)).long().cuda()]
                fg_rois_per_im = rois_per_im
            elif bg_inds.numel() > 0:
                to_replace = bg_inds.numel() < rois_per_im
                bg_inds = bg_inds[torch.from_numpy(
                    npr.choice(np.arange(0, bg_inds.numel()),
                               size=int(rois_per_im),
                               replace=to_replace)).long().cuda()]
                fg_rois_per_im = 0
            else:
                import pdb
                pdb.set_trace()

            # # ====================rectify========================
            # # Compute number of background RoIs to take from this image
            # # (guarding against there being fewer than desired)
            # bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
            # bg_rois_per_this_image = min(bg_rois_per_this_image,
            # bg_inds.size(0))
            # # Sample background regions without replacement
            # if bg_inds.size(0) > 0:
            #   bg_inds = bg_inds[torch.from_numpy(
            #     npr.choice(np.arange(0, bg_inds.numel()),
            # size=int(bg_rois_per_this_image), replace=False)).long().cuda()]

            # The indices that we're selecting (both fg and bg)
            if not isinstance(fg_inds, torch.cuda.LongTensor):
                print(fg_inds, type(fg_inds))
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            # Select sampled values from various arrays:
            label = label[keep_inds].contiguous()
            # Clamp labels for the background RoIs to 0
            label[int(fg_rois_per_im):] = 0
            roi = al_rois[keep_inds].contiguous()
            roi_score = al_scores[keep_inds].contiguous()
            tr_param = tr_param[keep_inds].contiguous()

            p_label = None
            if gt_box.size(1) > 5:
                p_label = gt_box[gt_assignment, [5]]
                p_label = p_label[keep_inds].contiguous()
                p_label[fg_rois_per_im:] = num_pid

            bbox_target_data = _compute_targets(
                roi[:, 1:5].data,
                gt_box[gt_assignment[keep_inds]][:, :4].data, label.data)

            bbox_tar, bbox_in_weights = _get_bbox_regression_labels(
                bbox_target_data, num_classes)

            return label, roi, roi_score, bbox_tar, bbox_in_weights, p_label, \
                   tr_param

        # ##################################################################
        # ========================Begin this method=========================
        # ##################################################################

        _num_classes = 2
        all_rois = rpn_rois
        all_scores = rpn_scores

        # Include ground-truth boxes in the set of candidate rois
        zeros = rpn_rois.data.new(gt_boxes.size(0), 1)
        all_rois = torch.cat(
            (torch.cat((zeros, gt_boxes.data[:, :4]), 1),
             all_rois), 0)
        # this may be a mistake, but all_scores is redundant anyway
        all_scores = torch.cat((all_scores, zeros), 0)
        gt_trans_param = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        gt_trans_param = gt_trans_param.expand(gt_boxes.size(0), 6)
        trans_param = torch.cat(
            (gt_trans_param.cuda(), trans_param), 0)

        num_images = 1
        rois_per_image = self.config['train_batch_size'] / num_images
        fg_rois_per_image = int(round(
            self.config['train_fg_frac'] * rois_per_image))

        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, roi_scores, bbox_targets, bbox_inside_weights, \
        pid_label, trans_param = _sample_rois(
            all_rois, all_scores, trans_param, gt_boxes, fg_rois_per_image,
            rois_per_image, _num_classes, self.num_pid)

        rois = rois.view(-1, 5)
        assert rois.size(0) == 128
        labels = labels.view(-1, 1)
        bbox_targets = bbox_targets.view(-1, _num_classes * 4)
        bbox_inside_weights = bbox_inside_weights.view(-1, _num_classes * 4)
        bbox_outside_weights = (bbox_inside_weights > 0).float()
        pid_label = pid_label.view(-1, 1)
        labels = labels.long()
        pid_label = pid_label.long()

        returns = (rois, (labels, pid_label), trans_param,
                   (bbox_targets, bbox_inside_weights, bbox_outside_weights))

        return returns
