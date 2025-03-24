from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch
import os
import signal
import subprocess
import time

# Path to the system monitoring script
MONITOR_SCRIPT_PATH = "/home/appuser/scripts/finetuning/SystemMonitor.sh"

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

def main():
    process = start_monitoring()

    register_coco_instances("my_dataset_train", {}, "/home/appuser/dataset/updated_train.json", "")
    register_coco_instances("my_dataset_val", {}, "/home/appuser/dataset/updated_train.json", "")

    cfg = get_cfg()
    cfg.merge_from_file("/home/appuser/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = "/home/appuser/models/model_final_f10217.pkl"

    cfg.SOLVER.IMS_PER_BATCH = 4  # e.g., 2 GPUs x 2 images each
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 150000
    cfg.SOLVER.STEPS = [120000, 140000]
    cfg.SOLVER.GAMMA = 0.1
    cfg.TEST.EVAL_PERIOD = 25000

    cfg.OUTPUT_DIR = "/mnt/PhenomicsProjects/Detectron2/Apptainer/detectron2/detectron2_sandbox/home/appuser/models"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    terminate_monitoring(process)

if __name__ == "__main__":
    # Set environment variables for torch.distributed using Slurm
    if "SLURM_PROCID" in os.environ:
        proc_id = int(os.environ["SLURM_PROCID"])
        num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", 1))
        num_nodes = int(os.environ.get("SLURM_NNODES", 1))
        node_rank = int(os.environ.get("SLURM_NODEID", 0))

        hostnames = os.environ.get("SLURM_NODELIST", "127.0.0.1").split(',')
        os.environ["MASTER_ADDR"] = hostnames[0]
        os.environ["MASTER_PORT"] = "29500"  # Can be any free port

        launch(
            main,
            num_gpus_per_machine=num_gpus,
            num_machines=num_nodes,
            machine_rank=node_rank,
        )
    else:
        # Fallback for non-Slurm
        launch(main, num_gpus_per_machine=1)
