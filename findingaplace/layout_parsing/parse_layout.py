from detectron2.engine import DefaultPredictor
import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
from detectron2.utils.visualizer import Visualizer, ColorMode
import json
import torch


# MODULE FOR RUNNING A DETECTRON2 LAYOUT PARSER MODULE FINE-TUNED ON HISTORICAL ILLUSTRATED BOOKS FROM THE
# ILLUSTRATION ARCHIVE TRAINING AND INFERENCE CODE ADAPTED FROM THE OFFICIAL DETECTRON2 TUTORIAL
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=ZyAvNCJMmvFF

def get_prima_dicts(img_dir, annotations_file):
    """
    registers dataset in COCO format
    :param img_dir: folder of images
    :return: dictionary of dataset in COCO format
    """
    json_file = os.path.join(annotations_file)
    try:
        f = open(json_file)
    except:
        print(f'check that the variable {img_dir} is a directory of images containing an annotations.json file')
        return
    images = json.load(f)
    f.close()
    dataset_dicts = list()
    for item in images:
        record = dict()
        record["file_name"] = os.path.join(item['file_name'])
        record["image_id"] = item['id']
        record["height"] = item['height']
        record["width"] = item['width']
        dataset_dicts.append(record)
    return dataset_dicts


#class_map = {0: "caption", 1: "text", 2: "figure", 3: "title"}   # layout categories to infer

def load_categories_as_dict(categories_as_string):
    return {k: v for v, k in enumerate(categories_as_string)}


def write_to_dict(outputs, name, filename, labels_path, cats):
    """
    writes predictions for each image to dictionary
    :param outputs: model prediction
    :param filename: image filename
    :param labels_path: folder of prediction files in json format
    :return:
    """
    cat_dict = load_categories_as_dict(cats)
    categories = list()
    for cat in cat_dict:
        d = {"id": cat, "name": cat_dict[cat]}
        categories.append(d)
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy().tolist()    # predicted bounding boxes in list format
    classes = outputs["instances"].pred_classes.cpu().numpy().tolist()       # predicted categories in list format
    classes = [categories[i] for i in classes]
    new_boxes = list()
    for box in boxes: # for each predicted bounding box
        new_boxes.append([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])  # write bounding boxes coordinates in opencv format
    shapes = [{"label": classes[i], "points": new_boxes[i]} for i in range(len(classes))]    # save category and coordinates in dictionary format
    annos = {'filename': name, 'annotations': shapes}
    jsonobject = json.dumps(annos)
    with open(os.path.join(labels_path, filename.rsplit('.', 1)[0] + '.json'), 'w') as f:  # save predictions as json file
        f.write(jsonobject)
    f.close()


def configurations(device, model_path, name):
    """
    get model configurations
    :param device: either cpu or gpu, based on your device
    :param model_path: path to model weights
    :param name: name of dataset (chosen)
    :return: configurations
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TEST = (name + "_predict",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.WEIGHTS = os.path.join(model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    return cfg


def predict(annotations_file, test_path, labels_path, name, model_path, cats, output_path=False):
    """
    function for inference on dataset
    :param test_path: folder of images to parse
    :param labels_path: folder to store predictions
    :param name: name of dataset
    :param model_path: path to model weights
    :param device: either cpu or gpu
    :param output_path: path to store images with visualised predicted bounding boxes
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for d in ['predict']:
        DatasetCatalog.register(name + "_" + d, lambda d=d: get_prima_dicts(test_path, annotations_file))
    MetadataCatalog.get(name + "_predict").set(thing_classes=["caption", "text", "figure", "title"])
    MetadataCatalog.get(name + "_predict").set(thing_colors=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 255)])
    cfg = configurations(device, model_path, name)
    predictor = DefaultPredictor(cfg)
    balloon_metadata = MetadataCatalog.get(name + "_predict")
    dataset_dicts = get_prima_dicts(test_path, annotations_file)
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        # print(d["file_name"])
        name = d["file_name"].rsplit('\\', 1)[1]
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.SEGMENTATION
                       )
        write_to_dict(outputs, d['file_name'], name, labels_path, cats)
        if output_path != False:
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out = v.draw_dataset_dict(d)
            cv2.imwrite(os.path.join(output_path, name), out.get_image()[:, :, ::-1])
