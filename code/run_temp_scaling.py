import json
import logging
import os
import types

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig, instantiate
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
from detectron2.engine import default_argument_parser
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_model
from detectron2.modeling.meta_arch import GeneralizedRCNN, RetinaNet
from detectron2.modeling.meta_arch.fcos import FCOS
from torch import nn
from tqdm import tqdm

from evaluation import evaluate_ece_from_coco_eval
from losses.temperature_scaling_inference import *
from utils.common_train import register_train_datasets, get_lookup_file
from utils.constants import WORKDIR, DEVICE, AREA_RNG
from utils.custom_evaluator import COCOEvaluator

logger = logging.getLogger("detectron2")


def get_setup_lazy(args, best_ckp_path, path):
    cfg = LazyConfig.load(os.path.join(path, f'config.yaml'))
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    cfg.train.init_checkpoint = best_ckp_path
    cfg.model.test_score_thresh = args.test_score_thresh

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    if 'coco' in cfg.dataloader.test.dataset.names:
        seed = path.split("/")[-1].split("_")[-1]
        val_dataset_name = f"coco_val_{seed}"
        cfg.dataloader.test.dataset = L(get_detection_dataset_dicts)(names=val_dataset_name)
        val_loader = instantiate(cfg.dataloader.test)
    else:
        val_loader = instantiate(cfg.dataloader.test)
        val_dataset_name = cfg.dataloader.test.dataset.names

    return cfg, model, val_loader, val_dataset_name


def get_setup(args, best_ckp_path, path):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(path, f'config.yaml'))
    cfg.merge_from_list(args.opts)

    cfg.MODEL.WEIGHTS = best_ckp_path
    # Set score_threshold for RetinaNet and Faster RCNN
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.test_score_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.test_score_thresh

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    if 'coco' in cfg.DATASETS.TEST[0]:
        val_dataset_name = f"coco_val_{cfg.SEED}"
        val_loader = build_detection_test_loader(cfg, val_dataset_name)
    else:
        val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST)
        val_dataset_name = cfg.DATASETS.TEST[0]

    return cfg, model, val_loader, val_dataset_name


def evaluate_model(model, temp, val_dataloader, val_dataset_name, path_to_out_folder, file):
    if temp:
        model.temperature = temp
        if isinstance(model, GeneralizedRCNN):
            model.roi_heads.box_predictor.inference = types.MethodType(inference_rcnn, model)
        elif isinstance(model, RetinaNet):
            model.forward_inference = types.MethodType(forward_inference_retina, model)
        elif isinstance(model, FCOS):
            model.forward_inference = types.MethodType(forward_inference_fcos, model)

    model.eval()
    evaluator = COCOEvaluator(val_dataset_name, output_dir=path_to_out_folder)
    result = inference_on_dataset(model, val_dataloader, evaluator)
    ece = evaluate_ece_from_coco_eval(evaluator.coco_eval, file)

    file.write(f"MAP: \n {json.dumps(result['bbox'], indent=2)} \n")
    file.write(f"ECE: \n {json.dumps(ece, indent=2)} \n")

    return evaluator.coco_eval


def temp_scale_model(coco_eval, file):
    nll_criterion = nn.BCEWithLogitsLoss().to(DEVICE)

    scores, labels = collect_scores_and_labels(coco_eval)
    logits = inverse_sigmoid(scores)
    file.write(f"Before temperature scaling NLL = { nll_criterion(logits, labels).item()} \n")

    temp_values = torch.linspace(1e-4, 5, steps=10000)
    optim_temp = -1
    best_loss = torch.finfo(torch.float).max
    for temp in tqdm(temp_values):
        loss = nll_criterion((logits / temp), labels)
        if loss < best_loss:
            best_loss = loss
            optim_temp = temp

    file.write(f"'Optimal temperature: {optim_temp} \n")
    file.write(f"After temperature scaling NLL = { nll_criterion((logits / optim_temp), labels).item()} \n")

    return optim_temp


def collect_scores_and_labels(coco_eval):
    filtered_eval_imgs_all = [img for img in coco_eval.evalImgs if img and img['aRng'] == AREA_RNG[0]]
    scores = torch.tensor(np.concatenate([img['dtScores'] for img in filtered_eval_imgs_all])).unsqueeze(-1)
    # Array with binary values whether the prediction matched gt or not
    matched = torch.tensor(get_matched(filtered_eval_imgs_all, t=0)).unsqueeze(-1)

    return scores, matched


def get_matched(filtered_eval_imgs, t=0):
    matched = []
    for filtered_eval_img in filtered_eval_imgs:
        dt_matches = filtered_eval_img['dtMatches'][t]
        # Change all non-zero elements to 1
        dt_matches[dt_matches != 0] = 1
        matched.append(dt_matches)

    flat_matched = [item for sublist in matched for item in sublist]
    return flat_matched


def inverse_sigmoid(p):
    return torch.log(p/(1-p))


def process_folder(folder_name, args):
    lookup = get_lookup_file(folder_name)
    for out_folder in os.listdir(os.path.join(WORKDIR, folder_name)):
        out_folder = str(out_folder)
        print(out_folder)
        if out_folder.startswith('output'):
            path_to_out_folder = os.path.join(WORKDIR, folder_name, out_folder)
            best_ckp_path = os.path.join(path_to_out_folder, lookup[out_folder[15:]]['ckp'])

            setup_fcn = get_setup_lazy if 'fcos' in out_folder else get_setup
            cfg, model, val_dataloader, val_dataset_name = setup_fcn(args, best_ckp_path, path_to_out_folder)

            file = open(f"{path_to_out_folder}/temp_scale_val_metrics.txt", 'w')
            file.write("Before temperature scaling \n")
            coco_eval = evaluate_model(model, None, val_dataloader, val_dataset_name, path_to_out_folder, file)
            temp = temp_scale_model(coco_eval, file)
            temp_file = open(f"{path_to_out_folder}/optimal_temperature.txt", 'w')
            temp_file.write(str(temp.item()))
            file.write(f"After temperature scaling with t={temp}\n")
            evaluate_model(model, temp, val_dataloader, val_dataset_name, path_to_out_folder, file)


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument(
        "--test_score_thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--folders",
        nargs='+',
        default=['_models_cityscapes', '_models_voc', '_models_coco']
    )

    register_train_datasets()
    args = parser.parse_args()
    for folder in args.folders:
        process_folder(folder, args)
