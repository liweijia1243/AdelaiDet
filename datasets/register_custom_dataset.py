from detectron2.data import DatasetCatalog, MetadataCatalog
import os

from detectron2.data.datasets import load_coco_json

CLASS_NAMES = ["dash", 'solid']
DATASET_ROOT = './my_data/'
ANN_ROOT = DATASET_ROOT
TRAIN_PATH = ANN_ROOT
VAL_PATH = ANN_ROOT
TRAIN_JSON = os.path.join(ANN_ROOT, 'result.json')
VAL_JSON = os.path.join(ANN_ROOT, 'val.json')
PREDEFINED_SPLITS_DATASET = {
    "cloud_map_train": (TRAIN_PATH, TRAIN_JSON),
    "cloud_map_val": (VAL_PATH, VAL_JSON),
}

def plain_register_dataset():
    # 训练集
    DatasetCatalog.register(
        "cloud_map_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("cloud_map_train").set(thing_classes=CLASS_NAMES,
                                            evaluator_type='coco',
                                            json_file=TRAIN_JSON,
                                            image_root=TRAIN_PATH)
    DatasetCatalog.register(
        "cloud_map_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("cloud_map_val").set(thing_classes=CLASS_NAMES,
                                          evaluator_type='coco',
                                          json_file=VAL_JSON,
                                          image_root=VAL_PATH)
                                          
plain_register_dataset()