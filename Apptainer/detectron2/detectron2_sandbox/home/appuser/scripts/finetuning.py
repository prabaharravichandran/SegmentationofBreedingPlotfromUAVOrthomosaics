from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import os

register_coco_instances("my_dataset_train", {}, "/home/appuser/dataset/updated_train.json", "")
register_coco_instances("my_dataset_val", {}, "/home/appuser/dataset/updated_train.json", "")

cfg = get_cfg()
cfg.merge_from_file("/home/appuser/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)

# Set number of classes in your dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # e.g., 1 if you have one class

cfg.DATALOADER.NUM_WORKERS = 2

# Set weights for pre-trained model (COCO or your custom model)
cfg.MODEL.WEIGHTS = "/home/appuser/models/model_final_f10217.pkl"  # Start from ImageNet weights or COCO weights

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # Learning rate, tune as needed
cfg.SOLVER.MAX_ITER = 20000   # Adjust iterations as per dataset size
cfg.SOLVER.STEPS = [15000, 18000]  # LR decays at these iterations
cfg.SOLVER.GAMMA = 0.1  # Multiply LR by 0.1 at decay points
cfg.TEST.EVAL_PERIOD = 5000

cfg.OUTPUT_DIR = "/mnt/PhenomicsProjects/Detectron2/Apptainer/detectron2/detectron2_sandbox/home/appuser/models"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
