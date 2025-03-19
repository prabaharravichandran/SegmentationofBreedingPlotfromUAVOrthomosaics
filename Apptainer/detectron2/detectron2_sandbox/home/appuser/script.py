from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2

setup_logger()

cfg = get_cfg()
cfg.merge_from_file("/home/appuser/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Set custom weights
cfg.MODEL.WEIGHTS = "/home/appuser/models/model_final_f10217.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

# Optional: Set number of classes if not using COCO
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
# MetadataCatalog.get("my_dataset").thing_classes = ["class_name"]

# Create predictor
predictor = DefaultPredictor(cfg)

# Load image
input_path = "/mnt/PhenomicsProjects/Detectron2/Apptainer/detectron2/detectron2_sandbox/home/appuser/prediction/image.jpg"
output_path = "/mnt/PhenomicsProjects/Detectron2/Apptainer/detectron2/detectron2_sandbox/home/appuser/prediction/output.jpg"

im = cv2.imread(input_path)
outputs = predictor(im)

# Visualization
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else "__unused__"), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Save the image with predictions
result_img = out.get_image()[:, :, ::-1]
cv2.imwrite(output_path, result_img)

print(outputs)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

print(f"Saved prediction overlay to {output_path}")
