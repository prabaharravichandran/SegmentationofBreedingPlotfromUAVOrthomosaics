import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from segment_anything import sam_model_registry

# --------------- PARAMETERS ---------------
# Set target_size to (1024,1024) to force fixed-size training.
target_size = (1024, 1024)

# --------------- 1. Define Transforms ---------------
image_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize(target_size, interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# --------------- 2. Set Up the Dataset ---------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        # Resize images to target size
        image = image.resize(target_size)
        mask = mask.resize(target_size)
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

# Adjust these directories to your dataset locations
images_dir = "/mnt/PhenomicsProjects/Detectron2/Datasets/breedingplots/output/images"
masks_dir = "/mnt/PhenomicsProjects/Detectron2/Datasets/breedingplots/output/masks"
dataset = SegmentationDataset(
    image_dir=images_dir,
    mask_dir=masks_dir,
    image_transform=image_transform,
    mask_transform=mask_transform
)

# For fixed-size images, we use the default collate (batch is a tensor)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# --------------- 3. Load SAM with Official Checkpoint ---------------
checkpoint_path = "/mnt/PhenomicsProjects/Detectron2/Apptainer/segmentanything/segmentanything_sandbox/home/ubuntu/model_checkpoints/sam_vit_h_4b8939.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
# Use the official SAM model; if using vit_h, change model_type to "vit_h" as needed.
model_type = "default"  # or "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

# --------------- 4. Freeze the Image Encoder ---------------
for param in sam.image_encoder.parameters():
    param.requires_grad = False

# --------------- 5. Define a Custom Segmentation Head ---------------
class SAMDecoder(nn.Module):
    def __init__(self, input_dim, output_channels=1):
        super().__init__()
        # A simple 1x1 convolution as an example segmentation head.
        self.conv = nn.Conv2d(input_dim, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

segmentation_head = SAMDecoder(input_dim=256, output_channels=1)

# --------------- 6. Combine SAM with the Segmentation Head ---------------
class SAMFineTuner(nn.Module):
    def __init__(self, sam_model, segmentation_head):
        super().__init__()
        self.sam = sam_model
        self.segmentation_head = segmentation_head

    def forward(self, images):
        # images: [batch, C, H, W]
        features = self.sam.image_encoder(images)  # [batch, 256, H/16, W/16]
        masks = self.segmentation_head(features)   # [batch, 1, H/16, W/16]
        # Upsample to original image size (1024 x 1024)
        masks = F.interpolate(masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return masks

model = SAMFineTuner(sam, segmentation_head)

# --------------- Set Device and Print CUDA Status ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if device.type == 'cuda':
    num_gpus = torch.cuda.device_count()
    print("CUDA is being used. Number of GPUs available:", num_gpus)
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")

# --------------- 7. Training Loop with Official-Style Checkpoint Saving ---------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
num_epochs = 500
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # [batch, 1, 1024, 1024]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save checkpoint every 5 epochs
    os.makedirs("/mnt/PhenomicsProjects/Detectron2/Apptainer/segmentanything/segmentanything_sandbox/home/ubuntu/model_checkpoints", exist_ok=True)
    if (epoch + 1) % 100 == 0:
        save_path = f"/mnt/PhenomicsProjects/Detectron2/Apptainer/segmentanything/segmentanything_sandbox/home/ubuntu/model_checkpoints/finetuned_sam_epoch{epoch+1}.pth"
        checkpoint = {
            "image_encoder": sam.image_encoder.state_dict(),
            "prompt_encoder": sam.prompt_encoder.state_dict(),
            "mask_decoder": segmentation_head.state_dict(),  # saved as 'mask_decoder'
        }
        torch.save(checkpoint, save_path)
        print(f"Saved official-style checkpoint to {save_path}")

