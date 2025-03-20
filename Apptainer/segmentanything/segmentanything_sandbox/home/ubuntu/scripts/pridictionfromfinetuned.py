import torch
import cv2
import numpy as np
from segment_anything import sam_model_registry
import torch.nn as nn

# Define your custom decoder (must match fine-tuning)
class SAMDecoder(nn.Module):
    def __init__(self, input_dim=256, output_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Initialize original SAM model
model_type = "vit_h"
original_checkpoint = "/mnt/PhenomicsProjects/Detectron2/Apptainer/segmentanything/segmentanything_sandbox/home/ubuntu/model_checkpoints/sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=original_checkpoint)

# Replace original decoder with your custom decoder
sam.mask_decoder = SAMDecoder(input_dim=256, output_channels=1)

# Load your fine-tuned weights correctly
fine_tuned_checkpoint = "/home/ubuntu/model_checkpoints/finetuned_sam_epoch1000.pth"
sam.load_state_dict(torch.load(fine_tuned_checkpoint))
sam.eval()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam.to(device)

# Load and preprocess image
image_path = "/home/ubuntu/predictions/image.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_image = cv2.resize(image_rgb, (1024, 1024))
input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float() / 255
input_tensor = input_tensor.to(device)

# Run inference
with torch.no_grad():
    features = sam.image_encoder(input_tensor)
    output = sam.mask_decoder(features)
    output = torch.sigmoid(output)

# Post-process prediction
mask = output.squeeze().cpu().numpy()
mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
mask_binary = (mask_resized > 0.5).astype(np.uint8)

# Visualization (overlay mask on original image)
color = np.array([0, 255, 0], dtype=np.uint8)  # Green overlay
overlay = image_rgb.copy()
overlay[mask_binary == 1] = (0.5 * overlay[mask_binary == 1] + 0.5 * color).astype(np.uint8)

# Save result
overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
output_path = "/home/ubuntu/predictions/mask_overlay.jpg"
cv2.imwrite(output_path, overlay_bgr)

print(f"Mask overlay saved at {output_path}")
