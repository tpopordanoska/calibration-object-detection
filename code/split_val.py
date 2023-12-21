import json
import argparse
import funcy
from sklearn.model_selection import train_test_split


def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({'info': info, 'licenses': licenses, 'images': images,
                   'annotations': annotations, 'categories': categories}, coco, indent=2, sort_keys=True)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)

    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


def filter_images(images, annotations):
    annotation_ids = funcy.lmap(lambda i: int(i['image_id']), annotations)

    return funcy.lfilter(lambda a: int(a['id']) in annotation_ids, images)


parser = argparse.ArgumentParser(description='Splits COCO annotations file into training and test sets.')
parser.add_argument('--annotations', metavar='coco_annotations', type=str,
                    help='Path to COCO annotations file.')
parser.add_argument('--train', type=str, help='Where to store COCO training annotations')
parser.add_argument('--test', type=str, help='Where to store COCO test annotations')
parser.add_argument('--seed', default=5, type=int, help='Seed')
parser.add_argument('-s', dest='split', default=0.9, type=float, help="A percentage of a split; a number in (0, 1)")
args = parser.parse_args()


def main(args):

    with open(args.annotations, 'rt', encoding='UTF-8') as annotations:
        coco = json.load(annotations)
        info = coco['info'] if 'info' in coco else {}
        licenses = coco['licenses'] if 'licenses' in coco else {}
        images = coco['images']
        annotations = coco['annotations']
        categories = coco['categories']

        images_with_annotations = funcy.lmap(lambda a: int(a['image_id']), annotations)

        images = funcy.lremove(lambda i: i['id'] not in images_with_annotations, images)

        X_train, X_test = train_test_split(images, train_size=args.split)

        anns_train = filter_annotations(annotations, X_train)
        anns_test=filter_annotations(annotations, X_test)

        save_coco(args.train, info, licenses, X_train, anns_train, categories)
        save_coco(args.test, info, licenses, X_test, anns_test, categories)

        print("Saved {} entries in {} and {} in {}".format(len(anns_train), args.train, len(anns_test), args.test))


if __name__ == "__main__":
    main(args)
