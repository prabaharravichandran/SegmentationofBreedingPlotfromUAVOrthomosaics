# Training SegmentAnything & Detectron2 models for segmenting breeding plots in UAV orthomosaics

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Static Badge](https://img.shields.io/badge/Proudly-Canadian-FF0000.svg)

This project involves developing a custom Detectron2 model to detect breeding plots in orthomosaics. The workflow incorporates Label Studio for image annotation, which integrates with a PostgreSQL for data management.

<p align="center">
  <img src="https://prabahar.s3.ca-central-1.amazonaws.com/static/articles/detectron2_test_1.jpg" alt="Image Segmentation with Detectron2" width="100%">
  <br>
  <em>Figure 1: Segmentation with fine-tuned Detectron2</em>
</p>

<p align="center">
  <img src="https://prabahar.s3.ca-central-1.amazonaws.com/static/articles/segmentanything_test_1_5000.jpg" alt="Image Segmentation with SegmentAnything" width="100%">
  <br>
  <em>Figure 2: Segmentation with fine-tuned SegmentAnything</em>
</p>

Build apptainers for training Detectron2,

```bash
sbatch make.sbatch
```
for the first time, I do not have any dependencies installed,

run the writable container

salloc

```bash
salloc --job-name=TrainingSAM \
       --partition=gpu_a100 \
       --account=aafc_phenocart__gpu_a100 \
       --nodes=1 \
       --cpus-per-task=4 \
       --mem-per-cpu=128000M \
       --gres=gpu:4 \
       --qos=low \
       --time=48:00:00
```


```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
export TMPDIR=/mnt/cache
apptainer shell \
   --writable \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   segmentanything_sandbox
```

```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
export TMPDIR=/mnt/cache
apptainer shell \
   --writable \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   detectron2_sandbox
```


Once you are in, activate the environment and install the dependencies,

```bash
. /home/venv/bin/activate
```

Install nvidia-cuda-toolkit,

```bash
    # NVIDIA CUDA Toolkit "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb

    apt-get update
    apt-get install -y nvidia-cuda-toolkit
````

Let's install detectron2;

```bash
python -m pip install detectron2==0.6 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```


writable container

```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
apptainer shell \
   --writable \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   detectron2_sandbox/
```

```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
apptainer shell \
   --writable \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   segmentanything_sandbox/
```

writable container


```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
export TMPDIR=/mnt/cache
apptainer shell \
   --nv \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   segmentanything_sandbox/
```
Setup Postgres

```bash
./configure --prefix=/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql
make
make install

/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/initdb -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data
/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/pg_ctl -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data -l logfile start -o "-p 5433"
/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/pg_ctl -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data reload


kill $(pgrep postgres)

stop
/gpfs/fs7/aafc/phenocart/IDEs/Postgres/pgsql/bin/pg_ctl -D /gpfs/fs7/aafc/phenocart/IDEs/Postgres/data stop
# SemanticSegmentationwithDetectron2LabelStudioPostGRESDB
```

**PlotFinder** is a tool designed to segment agricultural plots in orthomosaic images using deep learning models. The goal is to accurately identify individual plots and generate shapefiles or polygon data that can be used in GIS applications for further analysis.

## Objectives
- Segment plots in orthomosaics using deep learning.
- Generate shapefiles or polygon data for integration with GIS tools.

## Workflow
1. **Raster Handling**
   - Load and visualize orthomosaic rasters with embedded geospatial metadata.

2. **Model Training**
   - Train deep learning models for plot segmentation and identification.

3. **Image Slicing & Processing**
   - Develop a method to slice large orthomosaic images.
   - Process sliced images through the trained deep learning model.
   - Stitch the segmented slices back into a unified output.

4. **Mask Generation**
   - Generate binary or multi-class masks for segmented plots.
   - Extract geospatial information from the masks.

5. **Shapefile Conversion**
   - Convert segmentation masks into shapefiles or polygon formats.
   - Enable downstream analysis in GIS platforms.

## Future Enhancements
- Optimize model inference speed for large-scale datasets.
- Add support for multiple plot shapes and sizes.
- Develop a GUI or web-based interface for easier use.

Fine-tuned Detectron2 model

```
https://prabahar.s3.ca-central-1.amazonaws.com/models/segmentation/detectron2/finetuned4plots_250324.pth
```