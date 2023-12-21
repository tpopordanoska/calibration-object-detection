import json
import os
import pathlib

from detectron2.data.datasets import register_coco_instances

from utils.constants import WORKDIR


def get_folder_name(args):
    if 'voc' in args.config_file.lower():
        return '_models_voc'
    elif 'cityscapes' in args.config_file.lower():
        return '_models_cityscapes'
    elif 'coco' in args.config_file.lower():
        return '_models_coco'


def get_lookup_file(folder_name):
    with open(f'{os.path.join(WORKDIR, f"{folder_name}.json")}', 'r') as file:
        return json.load(file)


def register_train_datasets():
    data_loc = pathlib.Path(os.getenv("DETECTRON2_DATASETS"))
    for seed_ds in [5, 10, 21]:
        for split_name in ["train", "val"]:
            json_loc_cityscapes = data_loc / "cityscapes" / "splits" / f"instances_cityscapes_{split_name}_{seed_ds}.json"
            register_coco_instances(f"cityscapes_{split_name}_{seed_ds}", {}, json_loc_cityscapes, data_loc / "cityscapes")
            pascalvoc_loc = data_loc / "VOCdevkit"
            json_loc_voc = pascalvoc_loc / "splits" / f"instances_voc_{split_name}_{seed_ds}.json"
            register_coco_instances(f"voc_{split_name}_{seed_ds}", {}, json_loc_voc, pascalvoc_loc)
            json_loc_coco = data_loc / "COCO" / "splits" / f"instances_coco_{split_name}_{seed_ds}.json"
            register_coco_instances(f"coco_{split_name}_{seed_ds}", {}, json_loc_coco, data_loc / "COCO" / "train2017")
