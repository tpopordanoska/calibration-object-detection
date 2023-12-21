import torch
from detectron2.layers import cat, cross_entropy
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from detectron2.structures import Instances, pairwise_iou
from detectron2.structures.boxes import Boxes
from detectron2.utils.events import get_event_storage
from fvcore.nn import sigmoid_focal_loss_jit
from torch.nn import functional as F

from losses.ece_kde import get_bandwidth, get_ece_kde
from utils.constants import IOU_THRS, EPS


def losses_rcnn(self, predictions, proposals):
    """
    Args:
        predictions: return values of :meth:`forward()`.
        proposals (list[Instances]): proposals that match the features that were used
            to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
            ``gt_classes`` are expected.

    Returns:
        Dict[str, Tensor]: dict of losses
    """
    scores, proposal_deltas = predictions

    # parse classification outputs
    gt_classes = (
        cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
    )
    _log_classification_stats(scores, gt_classes)

    # parse box regression outputs
    if len(proposals):
        proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
        assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
        # If "gt_boxes" does not exist, the proposals must be all negative and
        # should not be included in regression loss computation.
        # Here we just use proposal_boxes as an arbitrary placeholder because its
        # value won't be used in self.box_reg_loss().
        gt_boxes = cat(
            [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
            dim=0,
        )
    else:
        proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

    losses = {
        "loss_cal": get_cal_loss_rcnn(self, scores, proposals, proposal_boxes, gt_boxes, gt_classes),
        "loss_cls": cross_entropy(scores, gt_classes, reduction="mean"),
        "loss_box_reg": self.roi_heads.box_predictor.box_reg_loss(
            proposal_boxes, gt_boxes, proposal_deltas, gt_classes
        ),
    }
    return {k: v * self.roi_heads.box_predictor.loss_weight.get(k, 1.0) for k, v in losses.items()}


def losses_retina(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.

        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = _dense_box_regression_loss(
            anchors,
            self.box2box_transform,
            pred_anchor_deltas,
            gt_boxes,
            pos_mask,
            box_reg_loss_type=self.box_reg_loss_type,
            smooth_l1_beta=self.smooth_l1_beta,
        )

        if self.cal_loss == 'tcd':
            tcd = get_tcd_orig_impl_retina(self, pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels_target, valid_mask, pos_mask)
            return {
                "loss_fcos_cls": (loss_cls / normalizer) + tcd,
                "loss_fcos_loc": loss_box_reg / normalizer,
            }

        return {
            "loss_cal": get_cal_loss_retina(self, pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels),
            "loss_cls": loss_cls / normalizer,
            "loss_box_reg": loss_box_reg / normalizer,
        }


def losses_fcos(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes, pred_centerness):
    """
    This method is almost identical to :meth:`RetinaNet.losses`, with an extra
    "loss_centerness" in the returned dict.
    """
    num_images = len(gt_labels)
    gt_labels = torch.stack(gt_labels)  # (N, R)

    pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
    num_pos_anchors = pos_mask.sum().item()
    get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
    normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 300)

    # classification and regression loss
    gt_labels_target = F.one_hot(gt_labels, num_classes=self.num_classes + 1)[
                       :, :, :-1
                       ]  # no loss for the last (background) class
    loss_cls = sigmoid_focal_loss_jit(
        torch.cat(pred_logits, dim=1),
        gt_labels_target.to(pred_logits[0].dtype),
        alpha=self.focal_loss_alpha,
        gamma=self.focal_loss_gamma,
        reduction="sum",
    )

    loss_box_reg = _dense_box_regression_loss(
        anchors,
        self.box2box_transform,
        pred_anchor_deltas,
        [x.tensor for x in gt_boxes],
        pos_mask,
        box_reg_loss_type="giou",
    )

    ctrness_targets = self.compute_ctrness_targets(anchors, gt_boxes)  # NxR
    pred_centerness = torch.cat(pred_centerness, dim=1).squeeze(dim=2)  # NxR
    ctrness_loss = F.binary_cross_entropy_with_logits(
        pred_centerness[pos_mask], ctrness_targets[pos_mask], reduction="sum"
    )

    if self.cal_loss == 'tcd':
        tcd = get_tcd_orig_impl_fcos(self, pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels)
        return {
            "loss_fcos_cls": (loss_cls / normalizer) + tcd,
            "loss_fcos_loc": loss_box_reg / normalizer,
            "loss_fcos_ctr": ctrness_loss / normalizer,
        }

    return {
        "loss_cal": get_cal_loss_retina(self, pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels),
        "loss_fcos_cls": loss_cls / normalizer,
        "loss_fcos_loc": loss_box_reg / normalizer,
        "loss_fcos_ctr": ctrness_loss / normalizer,
    }


def get_cal_loss_rcnn(self, scores, proposals, proposal_boxes, gt_boxes, gt_classes):
    cal_loss = torch.tensor(0.)
    if self.cal_loss == 'kde':
        assert self.can_cal_lambda + self.bin_cal_id_lambda + self.bin_cal_thresh_lambda > 0
        kde_ece_cls, kde_ece_identity, avg_kde_ece_thresh = get_kde_ece_rcnn(
            self, scores, proposal_boxes, gt_boxes, gt_classes,
        )
        cal_loss = self.can_cal_lambda * kde_ece_cls + self.bin_cal_id_lambda * kde_ece_identity + \
                   self.bin_cal_thresh_lambda * avg_kde_ece_thresh

    if self.cal_loss == 'tcd':
        cal_loss = get_tcd_rcnn(scores, proposals, gt_classes, self.roi_heads.box_predictor.num_classes)

    return cal_loss


def get_kde_ece_rcnn(self, scores, proposal_boxes, gt_boxes, gt_classes):
    device, can_cal_lambda, bin_cal_id_lambda, bin_cal_thresh_lambda, bin_cal_type = \
        self.device, self.can_cal_lambda, self.bin_cal_id_lambda, self.bin_cal_thresh_lambda, self.bin_cal_type

    kde_ece_cls, kde_ece_identity, avg_kde_ece_thresh = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
    if can_cal_lambda > 0:
        softmax_scores = torch.nn.functional.softmax(scores, dim=1)
        y_onehot = F.one_hot(gt_classes, num_classes=softmax_scores.shape[1]).to(torch.float32)
        bandwidth = get_bandwidth(softmax_scores, device=device)
        kde_ece_cls = get_ece_kde(softmax_scores, y_onehot, bandwidth, p=1, cal_type='canonical', device=device)

    if bin_cal_id_lambda > 0 or bin_cal_thresh_lambda > 0:
        pos_mask = (gt_classes >= 0) & (gt_classes < (scores.shape[1] - 1))
        sigmoid_scores = torch.sigmoid(scores[:, :-1])[pos_mask]
        # gt_boxes are matched with proposal_boxes, so we need the elementwise iou scores, or the diag of the pairwise
        iou_scores = pairwise_iou(Boxes(gt_boxes[pos_mask]), Boxes(proposal_boxes[pos_mask])).diag()
        gt_classes = gt_classes[pos_mask]

        if bin_cal_type == 'top_label':
            confidences, predictions = torch.max(sigmoid_scores, dim=1)
            confidences = confidences.unsqueeze(-1)
            matched_labels = (predictions == gt_classes)
            kde_ece_identity, avg_kde_ece_thresh = get_cal(
                confidences, iou_scores, matched_labels, device, bin_cal_id_lambda, bin_cal_thresh_lambda
            )
        elif bin_cal_type == 'marginal':
            cat_kde_ece_identity, cat_kde_ece_thresh = [], []
            # Considering a one-vs-all classifier, all predicted bounding box have label=cat_idx
            matched_labels = torch.tensor([True] * len(gt_classes))
            for cat_idx in range(sigmoid_scores.shape[-1]):
                confidences = sigmoid_scores[:, cat_idx].unsqueeze(-1)
                kde_ece_identity, avg_kde_ece_thresh = get_cal(
                    confidences, iou_scores, matched_labels, device, bin_cal_id_lambda, bin_cal_thresh_lambda
                )
                cat_kde_ece_identity.append(kde_ece_identity)
                cat_kde_ece_thresh.append(avg_kde_ece_thresh)
            kde_ece_identity = torch.mean(torch.stack(cat_kde_ece_identity))
            avg_kde_ece_thresh = torch.mean(torch.stack(cat_kde_ece_thresh))

    return kde_ece_cls, kde_ece_identity, avg_kde_ece_thresh


def get_cal(confidences, iou_scores, matched_labels, device, bin_cal_id_lambda, bin_cal_thresh_lambda):
    kde_ece_identity, avg_kde_ece_thresh = torch.tensor(0.), torch.tensor(0.)

    f = torch.clamp(confidences, min=EPS, max=1 - EPS)
    iou_scores = torch.clamp(iou_scores, min=EPS, max=1 - EPS)
    bandwidth = get_bandwidth(f, device=device)
    matched_labels = matched_labels.to(iou_scores.device)

    if bin_cal_id_lambda > 0:
        kde_ece_identity = get_ece_kde(f, iou_scores, bandwidth, p=1, cal_type='binary', device=device)

    if bin_cal_thresh_lambda > 0:
        kde_ece_thresh = []
        for iou in IOU_THRS:
            matched = torch.where(
                (iou_scores >= iou) & matched_labels,
                torch.ones_like(iou_scores), torch.zeros_like(iou_scores)
            )
            kde_ece_thresh.append(get_ece_kde(f, matched, bandwidth, p=1, cal_type='binary', device=device))
        avg_kde_ece_thresh = torch.mean(torch.stack(kde_ece_thresh))

    return kde_ece_identity, avg_kde_ece_thresh


def get_cal_loss_retina(self, pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels):
    cal_loss = torch.tensor(0.)
    num_images = len(gt_labels)

    if self.cal_loss == 'kde':
        assert self.bin_cal_id_lambda + self.bin_cal_thresh_lambda > 0
        kde_ece_identity, avg_kde_ece_thresh = get_kde_ece_retina(
            self, anchors, pred_logits, pred_anchor_deltas, gt_labels, gt_boxes, num_images
        )
        cal_loss = self.bin_cal_id_lambda * kde_ece_identity + self.bin_cal_thresh_lambda * avg_kde_ece_thresh

    if self.cal_loss == 'tcd':
        cal_loss = get_tcd_retina(
            pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels, self.num_classes, self.box2box_transform
        )

    return cal_loss


def get_kde_ece_retina(self, anchors, pred_logits, pred_anchor_deltas, gt_labels, gt_boxes, num_images):
    num_classes, transform_fcn, bin_cal_id_lambda, bin_cal_thresh_lambda, bin_cal_type, device = \
        self.num_classes, self.box2box_transform, self.bin_cal_id_lambda, self.bin_cal_thresh_lambda, self.bin_cal_type, self.device

    kde_ece_identity, avg_kde_ece_thresh = torch.tensor(0.), torch.tensor(0.)
    if bin_cal_id_lambda > 0 or bin_cal_thresh_lambda > 0:
        if bin_cal_type == 'top_label':
            sample_pred_scores, sample_pred_classes, sample_pred_boxes, sample_gt_labels, sample_gt_boxes = get_samples(
                anchors, pred_logits, pred_anchor_deltas, gt_labels, gt_boxes, num_images, num_classes, transform_fcn
            )
            if not isinstance(sample_gt_boxes, Boxes):
                sample_gt_boxes = Boxes(sample_gt_boxes)
            iou_scores = pairwise_iou(sample_gt_boxes, Boxes(sample_pred_boxes)).diag()
            matched_labels = (sample_pred_classes == sample_gt_labels)
            kde_ece_identity, avg_kde_ece_thresh = get_cal(
                sample_pred_scores, iou_scores, matched_labels, device, bin_cal_id_lambda, bin_cal_thresh_lambda
            )
        elif bin_cal_type == 'marginal':
            cat_kde_ece_identity, cat_kde_ece_thresh = [], []
            for cat_idx in range(num_classes):
                sample_pred_scores, sample_pred_classes, sample_pred_boxes, sample_gt_labels, sample_gt_boxes = get_samples(
                    anchors, pred_logits, pred_anchor_deltas, gt_labels, gt_boxes, num_images, num_classes, transform_fcn, cat_idx
                )
                if not isinstance(sample_gt_boxes, Boxes):
                    sample_gt_boxes = Boxes(sample_gt_boxes)
                iou_scores = pairwise_iou(sample_gt_boxes, Boxes(sample_pred_boxes)).diag()
                matched_labels = torch.tensor([True] * len(sample_gt_labels))
                kde_ece_identity, avg_kde_ece_thresh = get_cal(
                    sample_pred_scores, iou_scores, matched_labels, device, bin_cal_id_lambda, bin_cal_thresh_lambda
                )
                cat_kde_ece_identity.append(kde_ece_identity)
                cat_kde_ece_thresh.append(avg_kde_ece_thresh)
            kde_ece_identity = torch.mean(torch.stack(cat_kde_ece_identity))
            avg_kde_ece_thresh = torch.mean(torch.stack(cat_kde_ece_thresh))

    return kde_ece_identity, avg_kde_ece_thresh


def get_samples(anchors, pred_logits, pred_anchor_deltas, gt_labels, gt_boxes, num_images, num_classes, transform_fcn, cat_idx=None):
    max_topk = min(num_images * 500, 8000)
    sampled_pred_scores, sampled_pred_boxes, sampled_gt_boxes, sampled_pred_classes, sampled_gt_labels = [], [], [], [], []
    anchors = Boxes(type(anchors[0]).cat(anchors).tensor)
    for img_idx in range(num_images):
        gt_labels_i = gt_labels[img_idx]
        pos_mask = (gt_labels_i >= 0) & (gt_labels_i != num_classes)
        scores_i = torch.cat([torch.sigmoid(x[img_idx]) for x in pred_logits])[pos_mask]
        deltas_i = torch.cat([x[img_idx] for x in pred_anchor_deltas])[pos_mask]
        gt_boxes_i = gt_boxes[img_idx][pos_mask]
        if not isinstance(gt_boxes_i, Boxes):
            gt_boxes_i = Boxes(gt_boxes_i)
        if cat_idx:
            scores_i = scores_i[:, cat_idx].unsqueeze(-1)
        num_topk = min(int(max_topk / num_images), len(scores_i))
        topk_pred_scores, topk_pred_classes, topk_pred_boxes, topk_gt_labels, topk_gt_boxes = get_topk(
            anchors, scores_i, deltas_i, gt_labels_i, gt_boxes_i, num_topk, transform_fcn
        )
        sampled_pred_scores.append(topk_pred_scores)
        sampled_pred_classes.append(topk_pred_classes)
        sampled_pred_boxes.append(topk_pred_boxes)
        sampled_gt_labels.append(topk_gt_labels)
        sampled_gt_boxes.append(topk_gt_boxes)

    sampled_pred_scores = torch.cat(sampled_pred_scores).unsqueeze(-1)
    sampled_pred_classes = torch.cat(sampled_pred_classes)
    sampled_pred_boxes = torch.cat(sampled_pred_boxes)
    sampled_gt_labels = torch.cat(sampled_gt_labels)
    sampled_gt_boxes = torch.cat(sampled_gt_boxes)

    return sampled_pred_scores, sampled_pred_classes, sampled_pred_boxes, sampled_gt_labels, sampled_gt_boxes


def get_topk(anchors, pred_scores, pred_deltas, gt_labels, gt_boxes, how_many_topk, box2box_transform):
    # Alternatively:
    # max_pred_score, idx = torch.max(pred_scores, dim=1)
    # pred_scores, pred_scores_idx = torch.topk(max_pred_score, how_many_topk)
    # pred_classes = idx[pred_scores_idx]
    # pred_boxes = box2box_transform.apply_deltas(pred_deltas[pred_scores_idx], anchors.tensor[pred_scores_idx])

    topk_pred_scores, topk_pred_scores_idx = torch.topk(pred_scores.flatten(), how_many_topk)
    topk_pred_scores_idx_row = topk_pred_scores_idx // pred_scores.shape[1]
    topk_pred_classes = topk_pred_scores_idx % pred_scores.shape[1]

    topk_pred_boxes = box2box_transform.apply_deltas(pred_deltas[topk_pred_scores_idx_row],
                                                     anchors.tensor[topk_pred_scores_idx_row])
    topk_gt_labels = gt_labels[topk_pred_scores_idx_row]
    topk_gt_boxes = gt_boxes.tensor[topk_pred_scores_idx_row]
    topk_anchors = anchors.tensor[topk_pred_scores_idx_row]

    return topk_pred_scores, topk_pred_classes, topk_pred_boxes, topk_gt_labels, topk_gt_boxes


def get_tcd_orig_impl_fcos(self, pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels):
    pred_scores = torch.sigmoid(torch.cat(pred_logits, dim=1)).flatten(0, 1)  # or .reshape(-1, self.num_classes)
    gt_labels_target = F.one_hot(gt_labels, num_classes=self.num_classes + 1)[:, :, :-1].flatten(0, 1)
    d_cls = torch.mean(torch.abs(torch.mean(pred_scores, 0) - torch.mean(gt_labels_target.float(), 0)))

    gt_labels_flat = gt_labels.flatten()
    anchors = type(anchors[0]).cat(anchors).tensor

    pos_inds = (gt_labels_flat >= 0) & (gt_labels_flat != self.num_classes)
    pred_boxes_pos = torch.stack([
        self.box2box_transform.apply_deltas(k, anchors) for k in torch.cat(pred_anchor_deltas, dim=1)
    ]).flatten(0, 1)[pos_inds]
    gt_boxes_pos = torch.stack([gt_boxes[i].tensor for i in range(len(gt_boxes))]).flatten(0, 1)[pos_inds]
    iou_scores_pos = pairwise_iou(Boxes(gt_boxes_pos), Boxes(pred_boxes_pos)).diag()
    d_det = torch.mean(torch.abs(iou_scores_pos - torch.max(pred_scores[pos_inds], dim=1)[0]))

    return 0.5 * (d_cls + 0.1 * d_det)


def get_tcd_orig_impl_retina(self, pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels_target, valid_mask, pos_mask):
    pred_scores = torch.sigmoid(torch.cat(pred_logits, dim=1))
    d_cls = torch.mean(torch.abs(torch.mean(pred_scores[valid_mask], 0) - torch.mean(gt_labels_target.float(), 0)))

    anchors = type(anchors[0]).cat(anchors).tensor
    pred_boxes_pos = torch.stack([
        self.box2box_transform.apply_deltas(k, anchors) for k in torch.cat(pred_anchor_deltas, dim=1)
    ])[pos_mask]
    gt_boxes_pos = torch.stack([gt_boxes[i] for i in range(len(gt_boxes))])[pos_mask]
    iou_scores_pos = pairwise_iou(Boxes(gt_boxes_pos), Boxes(pred_boxes_pos)).diag()
    d_det = torch.mean(torch.abs(iou_scores_pos - torch.max(pred_scores[pos_mask], dim=1)[0]))

    return 0.5 * (d_cls + 0.1 * d_det)


def get_tcd_retina(pred_logits, anchors, pred_anchor_deltas, gt_boxes, gt_labels, num_classes, box2box_transform):
    pred_scores = torch.sigmoid(torch.cat(pred_logits, dim=1))
    anchors = Boxes(type(anchors[0]).cat(anchors).tensor)

    avg_diff = []
    for k in range(num_classes):
        scores_k = pred_scores[:, :, k]
        q_k = (gt_labels == k).to(int)
        diff = torch.abs(torch.mean(scores_k - q_k))
        avg_diff.append(diff)

    d_cls = torch.mean(torch.stack(avg_diff))

    d_det = []
    for img_idx in range(len(pred_scores)):
        scores_i = torch.cat([torch.sigmoid(x[img_idx]) for x in pred_logits])
        deltas_i = torch.cat([x[img_idx] for x in pred_anchor_deltas])
        gt_labels_i = gt_labels[img_idx]
        gt_boxes_i = gt_boxes[img_idx]

        pos_mask_i = (gt_labels_i >= 0) & (gt_labels_i != num_classes)
        if sum(pos_mask_i) == 0:
            continue
        pred_boxes_i = box2box_transform.apply_deltas(deltas_i[pos_mask_i], anchors.tensor[pos_mask_i])
        if not isinstance(gt_boxes_i, Boxes):
            gt_boxes_i = Boxes(gt_boxes_i)
        iou_scores_pos = pairwise_iou(gt_boxes_i[pos_mask_i], Boxes(pred_boxes_i)).diag()
        scores_i_pos = scores_i[pos_mask_i]
        scores_i_pos_max, _ = torch.max(scores_i_pos, dim=1)
        d_det.append(torch.mean(torch.abs(iou_scores_pos - scores_i_pos_max)))

    d_det = torch.mean(torch.stack(d_det))

    return 1/2 * (d_cls + d_det)


def get_tcd_rcnn(pred_logits, proposals, gt_classes, num_classes):
    pred_scores = torch.sigmoid(pred_logits[:, :-1])

    avg_diff_cls = []
    for k in range(num_classes):
        scores_k = pred_scores[:, k]
        q_k = (gt_classes == k).to(int)
        avg_diff_cls.append(torch.abs(torch.mean(scores_k - q_k)))
    d_cls = torch.mean(torch.stack(avg_diff_cls))

    avg_diff_det = []
    for i, img in enumerate(proposals):
        proposal_boxes_i = img.proposal_boxes
        gt_classes_i = img.gt_classes
        gt_boxes_i = img.gt_boxes
        pred_scores_i = pred_scores[i * len(gt_classes_i): (i+1) * len(gt_classes_i)]

        pos_mask = (gt_classes_i >= 0) & (gt_classes_i != num_classes)
        iou_scores_i = pairwise_iou(gt_boxes_i[pos_mask], proposal_boxes_i[pos_mask]).diag()
        pred_scores_max_i, _ = torch.max(pred_scores_i[pos_mask], dim=1)
        avg_diff_det.append(torch.mean(torch.abs(iou_scores_i - pred_scores_max_i)))
    d_det = torch.mean(torch.stack(avg_diff_det))

    return 1/2 * (d_cls + d_det)
