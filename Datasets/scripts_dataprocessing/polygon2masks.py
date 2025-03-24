import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Base directory where your images are stored (from Label Studio)
base_media = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/Detectron2/venv/lib/python3.12/site-packages/label_studio/core/settings/media"

# Output directory for saving the dataset (images and masks)
output_dir = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/Detectron2/Datasets/breedingplots/output"
output_images_dir = os.path.join(output_dir, "images")
output_masks_dir = os.path.join(output_dir, "masks")

# Input json file from Label Studio
JSON = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/Detectron2/Datasets/breedingplots/train.json"

def convert_points(points, width, height):
    """
    If the polygon points are given as percentages (all values <= 100),
    convert them to absolute pixel coordinates.
    """
    # Check the first point; if its maximum value is <= 100, assume percentages
    if points and max(points[0]) <= 100:
        return [(point[0] * width / 100, point[1] * height / 100) for point in points]
    else:
        return points

def create_mask(image_size, polygons):
    """
    Create a binary mask image from a list of polygon annotations.
    """
    mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        draw.polygon(poly, outline=1, fill=1)
    return np.array(mask) * 255

# Ensure output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_masks_dir, exist_ok=True)

# Load the exported JSON file from Label Studio
with open(JSON, 'r') as f:
    data = json.load(f)

for item in data:
    # Get the image path from the task's data (adjust key if needed)
    image_path = item.get('data', {}).get('image')
    if not image_path:
        continue  # Skip items without an image reference

    # Map the image path if it starts with "/data/"
    if image_path.startswith("/data/"):
        rel_path = os.path.relpath(image_path, "/data")
        image_path_mapped = os.path.join(base_media, rel_path)
    else:
        image_path_mapped = image_path

    try:
        # Open the image using the mapped path
        image = Image.open(image_path_mapped)
    except Exception as e:
        print(f"Failed to open image: {image_path_mapped} with error: {e}")
        continue

    width, height = image.size

    # Collect all polygon annotations for this image
    polygons = []
    for ann in item.get('annotations', []):
        for result in ann.get('result', []):
            if result.get('type') == 'polygonlabels':
                points = result.get('value', {}).get('points', [])
                if points:
                    # Convert points to pixel coordinates if needed
                    polygon = convert_points(points, width, height)
                    polygons.append(polygon)

    # Save the original image to the output "images" folder
    file_name = os.path.basename(image_path_mapped)
    image_output_path = os.path.join(output_images_dir, file_name)
    try:
        image.save(image_output_path)
    except Exception as e:
        print(f"Failed to save image {image_output_path}: {e}")

    if polygons:
        mask = create_mask((width, height), polygons)
        # Save mask with the same base name but as PNG
        mask_file_name = os.path.splitext(file_name)[0] + ".png"
        mask_output_path = os.path.join(output_masks_dir, mask_file_name)
        cv2.imwrite(mask_output_path, mask)
        print(f"Saved mask for {image_path_mapped} at {mask_output_path}")
    else:
        print(f"No polygon annotations found for {image_path_mapped}")
