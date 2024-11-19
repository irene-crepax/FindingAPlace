from detectron2.engine import DefaultTrainer, DefaultPredictor
# from google.colab.patches import cv2_imshow
import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
import json
from detectron2.structures import BoxMode


def get_prima_dicts(img_dir):
    json_file = os.path.join(img_dir, "annotations.json")
    f = open(json_file)
    imgs_anns = json.load(f)
    annos = imgs_anns['annotations']
    images = imgs_anns['images']
    dataset_dicts = list()
    for item in images:
        record = dict()
        record["file_name"] = os.path.join(img_dir, item['file_name'])
        record["image_id"] = item['id']
        record["height"] = item['height']
        record["width"] = item['width']
        anno = [ann for ann in annos if ann['image_id'] == item['id']]
        objs = list()
        for ann in anno:
            obj = {
                "bbox": ann['bbox'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann['segmentation'],
                "category_id": ann['category_id'],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def run_training(img_dir, name, classes):
    for d in ["train", "val"]:
        DatasetCatalog.register("ia_" + d,
                                lambda d=d: get_prima_dicts(os.path.join(img_dir, d)))
    for d in ["train", "val"]:
        MetadataCatalog.get(name + "_" + d).set(thing_classes=classes)

    cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (name + "_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []  # do not decay learning rate
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    balloon_metadata = MetadataCatalog.get(name + "_val")
    dataset_dicts = get_prima_dicts(os.path.join(img_dir, "val"))
    for d in dataset_dicts:
        im = cv2.imread(d["file_name"])
        print(d["file_name"])
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5
                       )

        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out = v.draw_dataset_dict(d)
        # cv2_imshow(out.get_image()[:, :, ::-1])
        cv2.imwrite('output_images/' + d["file_name"].rsplit('/', 1)[1], out.get_image()[:, :, ::-1])
        print('boxes: ')
        print(outputs["instances"].pred_boxes)
        print('classes: ')
        print(outputs["instances"].pred_classes)
