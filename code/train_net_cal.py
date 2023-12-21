#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import types
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA, build_model
from detectron2.modeling.meta_arch import GeneralizedRCNN, RetinaNet

from losses.calibration_losses import *
from utils.common_train import *


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        setattr(model, 'can_cal_lambda', cfg['CAN_CAL_LAMBDA'])
        setattr(model, 'bin_cal_id_lambda', cfg['BIN_CAL_ID_LAMBDA'])
        setattr(model, 'bin_cal_thresh_lambda', cfg['BIN_CAL_THRESH_LAMBDA'])
        setattr(model, 'bin_cal_type', cfg['BIN_CAL_TYPE'])
        setattr(model, 'cal_loss', cfg['CAL_LOSS'])

        if isinstance(model, GeneralizedRCNN):
            model.roi_heads.box_predictor.losses = types.MethodType(losses_rcnn, model)
        elif isinstance(model, RetinaNet):
            model.losses = types.MethodType(losses_retina, model)

        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    if args.from_pretrained:
        folder_name = get_folder_name(args)
        lookup = get_lookup_file(folder_name)
        if 'coco' in folder_name:
            key = args.config_file.split('/')[-1].split('.')[0] + f"_{str(cfg.SEED)}"
        else:
            key = args.config_file.split('/')[-1].split('.')[0]

        cfg.MODEL.WEIGHTS = os.path.join(WORKDIR, folder_name, lookup[key]['folder_name'], lookup[key]['ckp'])
        cfg.SOLVER.MAX_ITER = args.num_finetune_iter
        cfg.SOLVER.BASE_LR = 1e-4

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_train_datasets()

    cfg = setup(args)
    # Allow adding new keys
    cfg.set_new_allowed(True)
    cfg['CAN_CAL_LAMBDA'] = args.can_cal_lambda
    cfg['BIN_CAL_ID_LAMBDA'] = args.bin_cal_id_lambda
    cfg['BIN_CAL_THRESH_LAMBDA'] = args.bin_cal_thresh_lambda
    cfg['CAL_LOSS'] = args.cal_loss
    cfg['BIN_CAL_TYPE'] = args.bin_cal_type

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--bin_cal_type',
                        choices=['top_label', 'marginal'],
                        default='marginal',
                        help='The type of binary calibration: top label or marginal')
    parser.add_argument('--cal_loss',
                        choices=['kde', 'tcd'],
                        default='kde')
    parser.add_argument('--can_cal_lambda',
                        type=float,
                        default=0,
                        help='Lambda for canonical calibration')
    parser.add_argument('--bin_cal_id_lambda',
                        type=float,
                        default=0,
                        help='Lambda for binary calibration with identity link')
    parser.add_argument('--bin_cal_thresh_lambda',
                        type=float,
                        default=0,
                        help='Lambda for binary classification with threshold link')
    parser.add_argument('--from_pretrained',
                        type=str,
                        default='false')
    parser.add_argument('--num_finetune_iter',
                        type=int,
                        default=100)

    args = parser.parse_args()
    args.from_pretrained = True if args.from_pretrained.lower() == 'true' else False

    if 'SLURM_JOB_NUM_NODES' in os.environ:
        args.num_machines = int(os.environ['SLURM_JOB_NUM_NODES'])
    if 'SLURM_NODEID' in os.environ:
        args.machine_rank = int(os.environ['SLURM_NODEID'])

    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
