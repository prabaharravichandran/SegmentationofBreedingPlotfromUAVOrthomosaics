import os
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'

import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
import cv2
from tqdm import tqdm
import warnings

# Suppress torch meshgrid warning
import torch
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

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
cfg.MODEL.WEIGHTS = "/home/appuser/models/model_0049999.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

MetadataCatalog.get("my_dataset").thing_classes = ["plots"]
predictor = DefaultPredictor(cfg)

# -----------------------------
# Tile + Predict + Vectorize + GeoTIFF + JPG with tqdm
# -----------------------------
def tile_predict_vectorize(input_tiff, output_folder, tile_width=1024, tile_height=1024):
    with rasterio.open(input_tiff) as src:
        img_width, img_height = src.width, src.height
        os.makedirs(output_folder, exist_ok=True)

        shp_output_dir = os.path.join(output_folder, "shapefiles")
        geojson_output_dir = os.path.join(output_folder, "geojson")
        geotiff_output_dir = os.path.join(output_folder, "geotiff")
        geotiff_mask_output_dir = os.path.join(output_folder, "geotiff_with_mask")
        jpg_output_dir = os.path.join(output_folder, "jpg")
        vis_output_dir = os.path.join(output_folder, "visuals")

        for d in [shp_output_dir, geojson_output_dir, geotiff_output_dir, geotiff_mask_output_dir, jpg_output_dir, vis_output_dir]:
            os.makedirs(d, exist_ok=True)

        tile_num = 0
        total_tiles = ((img_height - 1) // tile_height + 1) * ((img_width - 1) // tile_width + 1)
        for top in tqdm(range(0, img_height, tile_height), desc="Processing Rows"):
            for left in range(0, img_width, tile_width):
                window_width = min(tile_width, img_width - left)
                window_height = min(tile_height, img_height - top)

                window = Window(left, top, window_width, window_height)
                transform = src.window_transform(window)

                tile_data = src.read(window=window)
                rgb_tile = np.transpose(tile_data[:3], (1, 2, 0))

                if rgb_tile.dtype != np.uint8:
                    rgb_tile = np.clip(rgb_tile, 0, 255).astype(np.uint8)

                outputs = predictor(rgb_tile)

                pred_mask = np.zeros((window_height, window_width), dtype=np.uint8)

                for inst in outputs["instances"].pred_masks:
                    mask = inst.cpu().numpy().astype(np.uint8)
                    pred_mask = np.logical_or(pred_mask, mask).astype(np.uint8)

                # Save polygons from mask
                mask_polygons = [
                    shape(geom)
                    for geom, value in shapes(pred_mask, transform=transform)
                    if value == 1
                ]

                if mask_polygons:
                    gdf = gpd.GeoDataFrame(geometry=mask_polygons, crs=src.crs)

                    # Save Shapefile
                    shp_filename = os.path.join(shp_output_dir, f"tile_{tile_num}.shp")
                    gdf.to_file(shp_filename)

                    # Save GeoJSON
                    geojson_filename = os.path.join(geojson_output_dir, f"tile_{tile_num}.geojson")
                    gdf.to_file(geojson_filename, driver="GeoJSON")

                # Save original tile as GeoTIFF
                geotiff_tile_path = os.path.join(geotiff_output_dir, f"tile_{tile_num}.tif")
                with rasterio.open(
                    geotiff_tile_path, 'w',
                    driver='GTiff',
                    height=window_height,
                    width=window_width,
                    count=tile_data.shape[0],
                    dtype=tile_data.dtype,
                    crs=src.crs,
                    transform=transform
                ) as dst:
                    dst.write(tile_data)

                # Save masked tile as GeoTIFF
                masked_tile_data = tile_data * pred_mask
                geotiff_mask_tile_path = os.path.join(geotiff_mask_output_dir, f"tile_{tile_num}_mask.tif")
                with rasterio.open(
                    geotiff_mask_tile_path, 'w',
                    driver='GTiff',
                    height=window_height,
                    width=window_width,
                    count=tile_data.shape[0],
                    dtype=tile_data.dtype,
                    crs=src.crs,
                    transform=transform
                ) as dst_mask:
                    dst_mask.write(masked_tile_data)

                # Save original tile as JPG
                jpg_tile_path = os.path.join(jpg_output_dir, f"tile_{tile_num}.jpg")
                cv2.imwrite(jpg_tile_path, cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2BGR))

                # Visualization
                v = Visualizer(rgb_tile[:, :, ::-1], MetadataCatalog.get("my_dataset"), scale=1.0)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                vis_image = out.get_image()[:, :, ::-1]

                vis_filename = os.path.join(vis_output_dir, f"tile_{tile_num}.jpg")
                cv2.imwrite(vis_filename, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

                tile_num += 1

        print(f"Processed {tile_num} tiles. Outputs saved in '{output_folder}'.")

# -----------------------------
# Usage
# -----------------------------
input_tiff = '/mnt/PhenomicsProjects/Detectron2/Datasets/orthomosaics/M3M_20240815_UFPS.tif'
output_folder = '/mnt/PhenomicsProjects/Detectron2/Outputs'

tile_predict_vectorize(input_tiff, output_folder, 1024, 1024)
