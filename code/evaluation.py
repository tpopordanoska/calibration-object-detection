import json
import logging
import os
import pathlib
import types

import numpy as np
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig, instantiate
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset
from detectron2.modeling import build_model
from detectron2.modeling.meta_arch import GeneralizedRCNN, RetinaNet
from detectron2.modeling.meta_arch.fcos import FCOS
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

from losses.ece_kde import get_ece_kde, get_bandwidth
from losses.temperature_scaling_inference import *
from utils.common import dump_pickle, create_folder
from utils.constants import *
from utils.custom_evaluator import COCOEvaluator

from netcal.metrics.confidence.ECE import ECE
from netcal_utils.helper import read_json, match_frames_with_groundtruth
from netcal_utils.features import get_features

logger = logging.getLogger("detectron2")


def register_datasets():
    data_loc = pathlib.Path(os.getenv("DETECTRON2_DATASETS"))
    json_loc = data_loc / "cityscapes" / "splits" / f"instances_cityscapes_test.json"
    register_coco_instances(f"cityscapes_test", {}, json_loc, data_loc / "cityscapes")
    json_loc_voc = data_loc / "VOCdevkit" / "splits" / f"instances_voc_test.json"
    register_coco_instances(f"voc_test", {}, json_loc_voc, data_loc / "VOCdevkit")


def get_save_best_ckp(path):
    best_map = 0
    best_ckp = -1
    for file in os.listdir(path):
        if str(file).startswith('events.out.tfevents'):
            event_acc = EventAccumulator(os.path.join(path, file))
            event_acc.Reload()
            for e in event_acc.Scalars('bbox/AP'):
                if e.value > best_map:
                    best_map = e.value
                    best_ckp = e.step
            break
    if event_acc.Scalars('bbox/AP')[-1].step == best_ckp:
        ckp = 'model_final.pth'
    else:
        ckp = f"model_{best_ckp:07d}.pth"
    with open(os.path.join(path, 'best_checkpoint.txt'), 'w') as txt_file:
        txt_file.write(ckp)

    return ckp


def get_setup_lazy(args, best_ckp, test_dataset_name, path):
    cfg = LazyConfig.load(os.path.join(path, f'config.yaml'))
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    cfg.train.init_checkpoint = os.path.join(path, best_ckp)
    cfg.model.test_score_thresh = args.test_score_thresh
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    cfg.dataloader.test.dataset = L(get_detection_dataset_dicts)(names=test_dataset_name)
    test_loader = instantiate(cfg.dataloader.test)

    return cfg, model, test_loader


def get_setup(args, best_ckp, test_dataset_name, path):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(path, f'config.yaml'))
    cfg.merge_from_list(args.opts)

    cfg.MODEL.WEIGHTS = os.path.join(path, best_ckp)
    # Set score_threshold for RetinaNet and Faster RCNN
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.test_score_thresh
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.test_score_thresh

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    test_loader = build_detection_test_loader(cfg, test_dataset_name)

    return cfg, model, test_loader


def get_save_coco_eval(args, path, best_ckp, dataset, file):
    test_dataset_name = "coco_2017_val" if dataset == 'coco' else f"{dataset}_test"
    setup_fcn = get_setup_lazy if 'fcos' in path else get_setup

    cfg, model, test_loader = setup_fcn(args, best_ckp, test_dataset_name, path)

    if args.eval_temp_model:
        with open(os.path.join(path, 'optimal_temperature.txt')) as f:
            temp = float(f.readlines()[0])
            file.write(f"Temperature scaling with t={str(temp)}\n")
        model.temperature = temp
        if isinstance(model, GeneralizedRCNN):
            model.roi_heads.box_predictor.inference = types.MethodType(inference_rcnn, model)
        elif isinstance(model, RetinaNet):
            model.forward_inference = types.MethodType(forward_inference_retina, model)
        elif isinstance(model, FCOS):
            model.forward_inference = types.MethodType(forward_inference_fcos, model)

    model.eval()
    evaluator = COCOEvaluator(test_dataset_name, output_dir=path)
    result = inference_on_dataset(model, test_loader, evaluator)

    pickle_name = 'coco_eval_temp.pkl' if args.eval_temp_model else 'coco_eval.pkl'
    dump_pickle(os.path.join(path, pickle_name), evaluator.coco_eval)
    file.write(f"MAP: \n {json.dumps(result['bbox'], indent=2)} \n")

    return evaluator.coco_eval, result['bbox']


def evaluate_ece_from_coco_eval(coco_eval, file, link='thresh'):
    cat_ids = np.unique([img['category_id'] for img in coco_eval.evalImgs if img])
    avg_ece_all, avg_ece_small, avg_ece_medium, avg_ece_large = [], [], [], []
    res_per_category = {}
    avg_ece50 = []
    avg_ece75 = []
    for cat_id in tqdm(cat_ids):
        cat_label = coco_eval.cocoGt.cats[cat_id]['name']
        avg_ece_per_category = []
        if link == 'id':
            ece_all = evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 0, IOU_50_THRESHOLD_IDX, link)
            avg_ece_per_category.append(ece_all)
            avg_ece_all.append(ece_all)
            avg_ece50.append(ece_all)
            avg_ece75.append(evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 0, IOU_75_THRESHOLD_IDX, link))
            avg_ece_small.append(evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 1, IOU_50_THRESHOLD_IDX, link))
            avg_ece_medium.append(evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 2, IOU_50_THRESHOLD_IDX, link))
            avg_ece_large.append(evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 3, IOU_50_THRESHOLD_IDX, link))
        elif link == 'thresh':
            for t in range(NUM_IOU_THRESHOLDS):
                ece_all = evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 0, t, link)
                if t == IOU_50_THRESHOLD_IDX:
                    avg_ece50.append(ece_all)
                elif t == IOU_75_THRESHOLD_IDX:
                    avg_ece75.append(ece_all)
                avg_ece_per_category.append(ece_all)
                avg_ece_all.append(ece_all)
                avg_ece_small.append(evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 1, t, link))
                avg_ece_medium.append(evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 2, t, link))
                avg_ece_large.append(evaluate_ece_at_cat_scale_and_iou(coco_eval, cat_id, 3, t, link))

        res_per_category[f"ECE-{cat_label}"] = np.mean(avg_ece_per_category)

    results = ({
        f'ECE': float(np.mean(avg_ece_all)),
        f'ECE50': float(np.mean(avg_ece50)),
        f'ECE75': float(np.mean(avg_ece75)),
        f'ECEs': float(np.mean(avg_ece_small)),
        f'ECEm': float(np.mean(avg_ece_medium)),
        f'ECEl': float(np.mean(avg_ece_large)),
    })

    results.update(res_per_category)
    file.write(f"ECE-{link}: \n {json.dumps(results, indent=2)} \n")

    return results


def evaluate_ece_at_cat_scale_and_iou(coco_eval, cat=0, scale=0, t=0, link='thresh'):
    filtered_eval_imgs_by_area = [img for img in coco_eval.evalImgs if img and img['aRng'] == AREA_RNG[scale]]
    filtered_eval_imgs = [img for img in filtered_eval_imgs_by_area if img['category_id'] == cat]
    scores = torch.tensor(np.concatenate([img['dtScores'] for img in filtered_eval_imgs]))
    dt_ignore = np.concatenate([img['dtIgnore'][t, :] for img in filtered_eval_imgs])
    filtered_scores = torch.tensor([score for (score, ignore) in zip(scores, dt_ignore) if not ignore]).unsqueeze(-1)

    if len(filtered_scores) == 0:
        return 0

    if link == 'thresh':
        # Array with binary values whether the prediction matched gt or not
        matched = get_matched(filtered_eval_imgs, t)
    elif link == 'id':
        # Array with the corresponding IoU of the match
        matched = get_matched_ious(coco_eval, filtered_eval_imgs, t)

    filtered_matches = torch.tensor([iou for (iou, ignore) in zip(matched, dt_ignore) if not ignore])

    if len(filtered_scores) > MAX_LEN_SCORES:
        eces = evaluate_split(filtered_scores, filtered_matches, [])
        return np.mean(eces)

    bandwidth = get_bandwidth(filtered_scores, device=DEVICE)
    return get_ece_kde(filtered_scores, filtered_matches, bandwidth, p=1, cal_type='binary', device=DEVICE)


def evaluate_split(scores, matches, eces):
    if len(scores) <= MAX_LEN_SCORES:
        bandwidth = get_bandwidth(scores, device=DEVICE)
        eces.append(get_ece_kde(scores, matches, bandwidth, p=1, cal_type='binary', device=DEVICE))
        return eces

    scores1, scores2 = split_in_two(scores)
    matches1, matches2 = split_in_two(matches)
    evaluate_split(scores1, matches1, eces)
    evaluate_split(scores2, matches2, eces)

    return eces


def split_in_two(array):
    split = len(array) // 2
    return array[:split], array[split:]


def get_matched(filtered_eval_imgs, t=0):
    matched = []
    for filtered_eval_img in filtered_eval_imgs:
        dt_matches = filtered_eval_img['dtMatches'][t].copy()
        # Change all non-zero elements (matched) to 1
        dt_matches[dt_matches != 0] = 1
        matched.append(dt_matches)

    flat_matched = [item for sublist in matched for item in sublist]
    return flat_matched


def get_matched_ious(coco_eval, filtered_eval_imgs, t=0):
    # There will be one unique category_id for all filtered_eval_imgs because of previous filtering
    cat_id = filtered_eval_imgs[0]['category_id']
    matched_ious = []
    for filtered_eval_img in filtered_eval_imgs:
        m_ious = []
        img_id = filtered_eval_img['image_id']
        gt_matches = filtered_eval_img['gtMatches'][t]
        dt_matches = filtered_eval_img['dtMatches'][t][:len(gt_matches)]
        gt_ids = filtered_eval_img['gtIds']
        dt_ids = filtered_eval_img['dtIds']
        dt_matches_idx = [gt_ids.index(i) if i != 0 else -1 for i in dt_matches]
        ious = coco_eval.ious[img_id, cat_id]
        for aranged_dt_idx, dt_match_idx in zip(np.arange(len(dt_matches_idx)), dt_matches_idx):
            if dt_match_idx == -1:
                m_ious.append(0)
            else:
                m_ious.append(ious[aranged_dt_idx, dt_match_idx])
        # Add trailing 0s for the remaining non-matched detections
        if len(dt_ids) > len(gt_ids):
            m_ious = np.append(m_ious, np.zeros(len(dt_ids) - len(gt_ids)))
        matched_ious.append(m_ious)

    flat_matched_ious = [item for sublist in matched_ious for item in sublist]

    return flat_matched_ious


def evaluate_single_folder(path_to_folder, save_folder, dataset, lambda_folder):
    results = {
        'rcnn': {
            'res_map': [],
            'res_ece_thresh': [],
            'res_ece_id': [],
            'res_dece': []
        },
        'retina': {
            'res_map': [],
            'res_ece_thresh': [],
            'res_ece_id': [],
            'res_dece': []
        },
        'fcos': {
            'res_map': [],
            'res_ece_thresh': [],
            'res_ece_id': [],
            'res_dece': []
        },
    }

    for out_folder in os.listdir(path_to_folder):
        print(out_folder)

        if 'fcos' in out_folder:
            res = results['fcos']
        elif 'retina' in out_folder:
            res = results['retina']
        elif 'rcnn' in out_folder:
            res = results['rcnn']

        if str(out_folder).startswith('output'):
            path_to_out_folder = os.path.join(path_to_folder, out_folder)
            best_ckp = get_save_best_ckp(path_to_out_folder)
            file = open(f"{path_to_out_folder}/metrics.txt", 'a')
            coco_eval, map = get_save_coco_eval(args, path_to_out_folder, best_ckp, dataset, file)
            res['res_map'].append(map)
            res['res_ece_thresh'].append(evaluate_ece_from_coco_eval(coco_eval, file, 'thresh'))
            res['res_ece_id'].append(evaluate_ece_from_coco_eval(coco_eval, file, 'id'))
            res['res_dece'].append(evaluate_dece(path_to_out_folder, dataset, file))

    for key in results:
        res_map = results[key]['res_map']
        res_ece_thresh = results[key]['res_ece_thresh']
        res_ece_id = results[key]['res_ece_id']
        res_dece = results[key]['res_dece']

        if len(res_map) == 0:
            continue
        num_seeds = len(res_map)
        result_map = {k: f"{sum(d[k] for d in res_map) / num_seeds} \pm "
                         f"{np.std([d[k] for d in res_map]) / np.sqrt(num_seeds)}"
                      for k in [k for k in res_map[0]]}
        result_ece_thresh = {k: f"{sum(d[k] for d in res_ece_thresh) / num_seeds} \pm "
                                f"{np.std([d[k] for d in res_ece_thresh]) / np.sqrt(num_seeds)}"
                             for k in [k for k in res_ece_thresh[0]]}
        result_ece_id = {k: f"{sum(d[k] for d in res_ece_id) / num_seeds} \pm "
                                f"{np.std([d[k] for d in res_ece_id]) / np.sqrt(num_seeds)}"
                             for k in [k for k in res_ece_id[0]]}
        result_dece = {k: f"{sum(d[k] for d in res_dece) / num_seeds} \pm "
                                f"{np.std([d[k] for d in res_dece]) / np.sqrt(num_seeds)}"
                             for k in [k for k in res_dece[0]]}

        path = create_folder(os.path.join('../workdir', save_folder))
        filename = f"{dataset}_{key}_{lambda_folder}_metrics_avg.txt" if lambda_folder else f"{dataset}_{key}_metrics_avg.txt"
        file = open(os.path.join(path, filename), 'a')
        if args.eval_temp_model:
            file.write(f"Temperature scaling \n")
        file.write(f"Test score thresh: {args.test_score_thresh} \n")
        file.write(f"MAP: \n {json.dumps(result_map, indent=2)} \n")
        file.write(f"ECE-thresh: \n {json.dumps(result_ece_thresh, indent=2)} \n")
        file.write(f"ECE-id: \n {json.dumps(result_ece_id, indent=2)} \n")
        file.write(f"D-ECE: \n {json.dumps(result_dece, indent=2)} \n")

        file.write(f"Overleaf format: \n")
        result_map = {k: f"${sum(d[k] for d in res_map) / num_seeds:.2f}_{{\pm "
                         f"{np.std([d[k] for d in res_map]) / np.sqrt(num_seeds):.2f}}}$"
                      for k in [k for k in res_map[0]]}
        result_ece_thresh = {k: f"${100 * sum(d[k] for d in res_ece_thresh) / num_seeds:.2f}_{{\pm "
                                f"{100 * np.std([d[k] for d in res_ece_thresh]) / np.sqrt(num_seeds):.2f}}}$"
                             for k in [k for k in res_ece_thresh[0]]}
        result_ece_id = {k: f"${100 * sum(d[k] for d in res_ece_id) / num_seeds:.2f}_{{\pm "
                                f"{100 * np.std([d[k] for d in res_ece_id]) / np.sqrt(num_seeds):.2f}}}$"
                             for k in [k for k in res_ece_id[0]]}
        result_dece = {k: f"${100 * sum(d[k] for d in res_dece) / num_seeds:.2f}_{{\pm "
                                f"{100 * np.std([d[k] for d in res_dece]) / np.sqrt(num_seeds):.2f}}}$"
                             for k in [k for k in res_dece[0]]}

        file.write(f"MAP: {result_map['AP']} & {result_map['AP50']} & {result_map['AP75']} & "
                   f"{result_map['APs']} & {result_map['APm']} & {result_map['APm']} \n")
        file.write(f"ECE-thresh: {result_ece_thresh['ECE']} & {result_ece_thresh['ECE50']} & {result_ece_thresh['ECE75']} & "
                   f"{result_ece_thresh['ECEs']} & {result_ece_thresh['ECEm']} & {result_ece_thresh['ECEm']} \n")
        file.write(f"ECE-id: {result_ece_id['ECE']} & {result_ece_id['ECE50']} & {result_ece_id['ECE75']} & "
                   f"{result_ece_id['ECEs']} & {result_ece_id['ECEm']} & {result_ece_id['ECEm']} \n")
        file.write(f"D-ECE: {result_dece['D-ECE']} & {result_dece['D-ECE50']} & {result_dece['D-ECE75']} \n")


def process_folder(folder, args):
    dataset = folder.split("_")[-1]
    if args.eval_lambdas:
        # It means there is a list of subfolders for the different lambdas
        for lambda_folder in os.listdir(os.path.join(WORKDIR, folder)):
            print("Lambda folder: ", lambda_folder)
            evaluate_single_folder(os.path.join(WORKDIR, folder, lambda_folder), args.save_folder, dataset, lambda_folder)
    else:
        evaluate_single_folder(os.path.join(WORKDIR, folder), args.save_folder, dataset, None)


def evaluate_dece(path_to_out_folder, dataset, file):
    test_dataset_name = "coco_2017_val" if dataset == 'coco' else f"{dataset}_test"
    ious = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # define D-ECE metric
    dece = ECE(bins=20, detection=True)

    filename = os.path.join(path_to_out_folder, f"coco_instances_results.json")
    # read frames and match the frames with the according ground-truth boxes.
    # frames is a list of dicts that for every image_id contains the category_ids, bboxes, scores and matched
    frames = read_json(filename, score_threshold=args.test_score_thresh)
    frames = match_frames_with_groundtruth(frames, test_dataset_name, ious)

    category_ids = np.unique(np.concatenate([d['category_ids'] for d in frames], axis=0))
    test_ids = np.unique([d['image_id'] for d in frames]).tolist()
    avg_dece = torch.zeros(len(ious))
    for category_id in category_ids:
        features, matched, img_ids = get_features(frames, category_id, [], ious, test_ids)
        for j, (iou, m) in enumerate(zip(ious, matched)):
            avg_dece[j] += dece.measure(features, m)

    avg_dece = avg_dece / len(category_ids)

    results = {
        'D-ECE': float(torch.mean(avg_dece).numpy())
    }
    for j, iou in enumerate(ious):
        results.update({f'D-ECE{int(iou*100)}': float(avg_dece[j].numpy())})

    file.write(f"D-ECE: \n {json.dumps(results, indent=2)} \n")

    return results


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
    parser.add_argument(
        '--eval_temp_model',
        type=str,
        default='false'
    )
    parser.add_argument(
        '--eval_lambdas',
        type=str,
        default='true'
    )
    parser.add_argument(
        '--save_folder',
        type=str,
        default='FINAL_RESULTS'
    )
    args = parser.parse_args()
    args.eval_temp_model = True if args.eval_temp_model.lower() == 'true' else False
    args.eval_lambdas = True if args.eval_lambdas.lower() == 'true' else False

    register_datasets()
    for folder in args.folders:
        process_folder(folder, args)
