#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import types

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.modeling.meta_arch.fcos import FCOS
from detectron2.utils import comm

from losses.calibration_losses import *
from utils.common_train import *

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)

    setattr(model, 'can_cal_lambda', cfg.can_cal_lambda)
    setattr(model, 'bin_cal_id_lambda', cfg.bin_cal_id_lambda)
    setattr(model, 'bin_cal_thresh_lambda', cfg.bin_cal_thresh_lambda)
    setattr(model, 'bin_cal_type', cfg.bin_cal_type)
    setattr(model, 'cal_loss', cfg.cal_loss)

    if isinstance(model, FCOS):
        model.losses = types.MethodType(losses_fcos, model)

    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    register_train_datasets()

    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    cfg.can_cal_lambda = args.can_cal_lambda
    cfg.bin_cal_id_lambda = args.bin_cal_id_lambda
    cfg.bin_cal_thresh_lambda = args.bin_cal_thresh_lambda
    cfg.bin_cal_type = args.bin_cal_type
    cfg.cal_loss = args.cal_loss

    if args.from_pretrained:
        folder_name = get_folder_name(args)
        lookup = get_lookup_file(folder_name)
        if 'coco' in folder_name:
            key = args.config_file.split('/')[-1].split('.')[0] + f"_{str(cfg.train.seed)}"
        else:
            key = args.config_file.split('/')[-1].split('.')[0]
        cfg.train.init_checkpoint = os.path.join(WORKDIR, folder_name, lookup[key]['folder_name'], lookup[key]['ckp'])
        cfg.train.max_iter = args.num_finetune_iter
        cfg.optimizer.lr = 1e-4

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


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
