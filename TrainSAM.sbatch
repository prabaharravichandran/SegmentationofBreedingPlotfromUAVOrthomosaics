#!/bin/bash -l

#SBATCH --job-name=TrainingSAM
#SBATCH --no-requeue
#SBATCH --partition=gpu_a100
#SBATCH --account=aafc_phenocart__gpu_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128000M  # Request 128 GB per CPU
#SBATCH --gres=gpu:4
#SBATCH --qos=low
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=prabahar.ravichandran@agr.gc.ca

# Change directory to where your Apptainer container is built (if needed)
cd /gpfs/fs7/aafc/phenocart/PhenomicsProjects/Detectron2/Apptainer/segmentanything

# Set up CUDA environment (if required by your workflow)
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1

# Instead of /mnt/cache (which is not writable), set TMPDIR to /tmp
export TMPDIR=/tmp

# The ORDENV_SETUP warning can be safely ignored if everything else is working.

# Use apptainer exec to run your commands inside the container.
apptainer exec \
    --nv \
    --fakeroot \
    --bind /gpfs/fs7/aafc/phenocart:/mnt \
    --contain \
    --no-home \
    segmentanything_sandbox \
    /bin/bash -c ". /home/venv/bin/activate && python /home/ubuntu/scripts/finetunesam.py"
