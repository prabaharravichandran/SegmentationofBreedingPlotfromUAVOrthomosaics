from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
import os
import signal
import subprocess
import time

# Path to the system monitoring script
MONITOR_SCRIPT_PATH = "/home/appuser/scripts/SystemMonitor.sh"

def start_monitoring():
    print("Starting system monitoring...")
    process = subprocess.Popen(
        ["bash", MONITOR_SCRIPT_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    time.sleep(2)
    retcode = process.poll()
    if retcode is not None:
        error_output = process.stderr.read().decode('utf-8').strip()
        print(f"Monitoring script failed to start (exit code: {retcode}).")
        print(f"stderr: {error_output}")
    else:
        print("Monitoring script started successfully, running in the background.")
    return process

def terminate_monitoring(process):
    print("Stopping system monitoring...")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

# Start monitoring
process = start_monitoring()

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
cfg.SOLVER.MAX_ITER = 50000   # Adjust iterations as per dataset size
cfg.SOLVER.STEPS = [40000, 45000]  # LR decays at these iterations
cfg.SOLVER.GAMMA = 0.1  # Multiply LR by 0.1 at decay points
cfg.TEST.EVAL_PERIOD = 25000

cfg.OUTPUT_DIR = "/mnt/PhenomicsProjects/Detectron2/Apptainer/detectron2/detectron2_sandbox/home/appuser/models"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Stop monitoring
terminate_monitoring(process)
