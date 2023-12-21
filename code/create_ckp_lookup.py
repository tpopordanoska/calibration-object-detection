import argparse
import json
import os

from utils.constants import WORKDIR


def process_folder(folder):
    lookup = {}
    for out_folder in os.listdir(os.path.join(WORKDIR, folder)):
        if str(out_folder).startswith('output'):
            path_to_out_folder = os.path.join(WORKDIR, folder, out_folder)
            with open(os.path.join(path_to_out_folder, 'best_checkpoint.txt')) as f:
                best_ckp = f.readlines()[0]
            lookup[out_folder[15:]] = {
                'folder_name': out_folder,
                'ckp': best_ckp
            }

    json_obj = json.dumps(lookup, indent=2)
    with open(f"{os.path.join(WORKDIR, folder)}.json", "w") as file:
        file.write(json_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders",
        nargs='+',
        default=['_models_cityscapes', '_models_voc', '_models_coco']
    )

    args = parser.parse_args()
    for folder in args.folders:
        process_folder(folder)
