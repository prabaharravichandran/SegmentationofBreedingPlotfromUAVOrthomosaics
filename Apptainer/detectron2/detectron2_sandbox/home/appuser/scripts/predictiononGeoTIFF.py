import rasterio
from rasterio.windows import Window
import os
import numpy as np
import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# -----------------------------
# Setup Detectron2 Predictor
# -----------------------------
setup_logger()

cfg = get_cfg()
cfg.merge_from_file("/home/appuser/detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = "/home/appuser/models/model_0019999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

MetadataCatalog.get("my_dataset").thing_classes = ["plots"]
predictor = DefaultPredictor(cfg)

# -----------------------------
# Tile + Predict + Mask Save
# -----------------------------
def tile_predict_geotiff(input_tiff, output_folder, tile_width=1024, tile_height=1024):
    with rasterio.open(input_tiff) as src:
        img_width, img_height = src.width, src.height
        meta = src.meta.copy()

        os.makedirs(output_folder, exist_ok=True)

        tile_num = 0
        for top in range(0, img_height, tile_height):
            for left in range(0, img_width, tile_width):
                window_width = min(tile_width, img_width - left)
                window_height = min(tile_height, img_height - top)

                window = Window(left, top, window_width, window_height)
                transform = src.window_transform(window)

                # Read RGB bands (assuming 3-band RGB)
                tile_data = src.read(window=window)  # shape: (bands, height, width)

                # Convert to HWC for OpenCV (for prediction)
                rgb_tile = np.transpose(tile_data[:3], (1, 2, 0))  # (height, width, 3)

                # Convert to uint8 if necessary
                if rgb_tile.dtype != np.uint8:
                    rgb_tile = np.clip(rgb_tile, 0, 255).astype(np.uint8)

                # Run Detectron2 prediction
                outputs = predictor(rgb_tile)

                # Create prediction mask (binary)
                pred_mask = np.zeros((window_height, window_width), dtype=np.uint8)

                for inst in outputs["instances"].pred_masks:
                    mask = inst.cpu().numpy()
                    pred_mask = np.logical_or(pred_mask, mask).astype(np.uint8)

                # Save visualized prediction (optional)
                vis_output_dir = os.path.join(output_folder, "visuals")
                os.makedirs(vis_output_dir, exist_ok=True)

                v = Visualizer(rgb_tile[:, :, ::-1], MetadataCatalog.get("my_dataset"), scale=1.0)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                vis_image = out.get_image()[:, :, ::-1]  # BGR

                vis_filename = os.path.join(vis_output_dir, f"tile_{tile_num}.jpg")
                cv2.imwrite(vis_filename, vis_image)

                # Update metadata with 1 extra band
                tile_meta = meta.copy()
                tile_meta.update({
                    "count": src.count + 1,
                    "height": window_height,
                    "width": window_width,
                    "transform": transform
                })

                # Save tile with prediction mask as last band
                tile_filename = os.path.join(output_folder, f"tile_{tile_num}.tif")
                with rasterio.open(tile_filename, 'w', **tile_meta) as dst:
                    # Write original bands
                    for i in range(1, src.count + 1):
                        dst.write(tile_data[i - 1], i)

                    # Write prediction mask
                    dst.write(pred_mask, src.count + 1)

                    # Band descriptions
                    for i in range(1, src.count + 1):
                        dst.set_band_description(i, f"Band_{i}")
                    dst.set_band_description(src.count + 1, "Prediction Mask")

                tile_num += 1

        print(f"Processed and saved {tile_num} tiles with predictions to '{output_folder}'")

# -----------------------------
# Usage
# -----------------------------
input_tiff = '/mnt/PhenomicsProjects/PlotFinder/Data/Orthomosaics/M3M_20240830_UFPS.tif'
output_folder = '/mnt/PhenomicsProjects/PlotFinder/Data/RGBRasters/PredictedTiles'

tile_predict_geotiff(input_tiff, output_folder, 1024, 1024)
