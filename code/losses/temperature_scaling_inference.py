from typing import List, Tuple

import torch
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.structures import ImageList, Instances
from torch import Tensor


def inference_rcnn(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):
    """
    Args:
        predictions: return values of :meth:`forward()`.
        proposals (list[Instances]): proposals that match the features that were
            used to compute predictions. The ``proposal_boxes`` field is expected.

    Returns:
        list[Instances]: same as `fast_rcnn_inference`.
        list[Tensor]: same as `fast_rcnn_inference`.
    """
    boxes = self.roi_heads.box_predictor.predict_boxes(predictions, proposals)
    pred_logits, pred_boxes = predictions
    if hasattr(self, 'temperature'):
        pred_logits /= self.temperature
        predictions = (pred_logits, pred_boxes)

    scores = self.roi_heads.box_predictor.predict_probs(predictions, proposals)
    image_shapes = [x.image_size for x in proposals]
    return fast_rcnn_inference(
        boxes,
        scores,
        image_shapes,
        self.roi_heads.box_predictor.test_score_thresh,
        self.roi_heads.box_predictor.test_nms_thresh,
        self.roi_heads.box_predictor.test_topk_per_image,
    )


def forward_inference_retina(
        self, images: ImageList, features: List[Tensor], predictions: List[List[Tensor]]
):
    pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
        predictions, [self.num_classes, 4]
    )
    if hasattr(self, 'temperature'):
        for feature_lvl in range(len(pred_logits)):
            pred_logits[feature_lvl] /= self.temperature

    anchors = self.anchor_generator(features)

    results: List[Instances] = []
    for img_idx, image_size in enumerate(images.image_sizes):
        scores_per_image = [x[img_idx].sigmoid_() for x in pred_logits]
        deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
        results_per_image = self.inference_single_image(
            anchors, scores_per_image, deltas_per_image, image_size
        )
        results.append(results_per_image)
    return results


def forward_inference_fcos(
    self, images: ImageList, features: List[Tensor], predictions: List[List[Tensor]]
):
    pred_logits, pred_anchor_deltas, pred_centerness = self._transpose_dense_predictions(
        predictions, [self.num_classes, 4, 1]
    )
    if hasattr(self, 'temperature'):
        for feature_lvl in range(len(pred_logits)):
            pred_logits[feature_lvl] /= self.temperature

    anchors = self.anchor_generator(features)

    results: List[Instances] = []
    for img_idx, image_size in enumerate(images.image_sizes):
        scores_per_image = [
            # Multiply and sqrt centerness & classification scores
            # (See eqn. 4 in https://arxiv.org/abs/2006.09214)
            torch.sqrt(x[img_idx].sigmoid_() * y[img_idx].sigmoid_())
            for x, y in zip(pred_logits, pred_centerness)
        ]
        deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
        results_per_image = self.inference_single_image(
            anchors, scores_per_image, deltas_per_image, image_size
        )
        results.append(results_per_image)
    return results
