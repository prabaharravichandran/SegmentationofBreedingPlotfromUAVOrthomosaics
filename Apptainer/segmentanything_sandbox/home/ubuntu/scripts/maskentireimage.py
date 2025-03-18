from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image using OpenCV
image = cv2.imread("/home/ubuntu/predictions/image.jpg")

# If needed, convert BGR (OpenCV default) to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam = sam_model_registry["default"](checkpoint="/home/ubuntu/model_checkpoints/sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)
# Now pass to generate()
masks = mask_generator.generate(image)

# Create a copy to overlay
overlay = image.copy()

# Assign random colors to each mask and overlay them
for mask_data in masks:
    mask = mask_data['segmentation']  # shape: (H, W)
    color = np.random.randint(0, 255, (3,), dtype=np.uint8)  # random color
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)

# Convert RGB to BGR for saving with OpenCV
overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

# Save image
cv2.imwrite("/home/ubuntu/predictions/mask_overlay.jpg", overlay_bgr)