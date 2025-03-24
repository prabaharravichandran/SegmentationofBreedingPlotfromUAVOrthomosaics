import os
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling

# -----------------------------
# Merge GeoTIFF Mask Tiles into One Mosaic
# -----------------------------
def merge_geotiff_masks(tiles_folder, output_path):
    tif_files = [
        os.path.join(tiles_folder, f) for f in os.listdir(tiles_folder)
        if f.endswith("_mask.tif")
    ]

    if not tif_files:
        print("No GeoTIFF mask tiles found in the specified folder.")
        return

    src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]

    mosaic, out_transform = merge(src_files_to_mosaic, resampling=Resampling.nearest)

    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in src_files_to_mosaic:
        src.close()

    print(f"Merged GeoTIFF saved at: {output_path}")

# -----------------------------
# Usage
# -----------------------------
tiles_folder = '/mnt/PhenomicsProjects/Detectron2/Outputs/geotiff_with_mask'
output_path = '/mnt/PhenomicsProjects/Detectron2/Outputs/mergedgeotiffs/merged_mask.tif'

merge_geotiff_masks(tiles_folder, output_path)
