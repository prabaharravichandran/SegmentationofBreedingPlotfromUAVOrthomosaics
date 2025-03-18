import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Import SAM's model registry (from the official Segment Anything codebase)
from segment_anything import sam_model_registry

# ------------------------------
# 1. Define Transforms for Images and Masks
# ------------------------------
# We force the images and masks to 1024x1024
image_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize((1024, 1024), interpolation=Image.NEAREST),
    transforms.ToTensor()
])


# ------------------------------
# 2. Set Up the Dataset
# ------------------------------
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

        # Ensure images are resized before applying transforms
        image = image.resize((1024, 1024))
        mask = mask.resize((1024, 1024))

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
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ------------------------------
# 3. Load SAM with the Provided Checkpoint
# ------------------------------
checkpoint_path = "/mnt/PhenomicsProjects/Detectron2/Apptainer/segmentanything_sandbox/home/ubuntu/model_checkpoints/sam_vit_h_4b8939.pth"
model_type = "default"  # Default SAM model

# Load the SAM model (this loads the image encoder, prompt encoder, and mask decoder)
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

# ------------------------------
# 4. Freeze the Image Encoder
# ------------------------------
for param in sam.image_encoder.parameters():
    param.requires_grad = False


# ------------------------------
# 5. Define a Custom Segmentation Head
# ------------------------------
class SAMDecoder(nn.Module):
    def __init__(self, input_dim, output_channels=1):
        super().__init__()
        # A simple 1x1 convolution as an example segmentation head.
        self.conv = nn.Conv2d(input_dim, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# With a 1024x1024 input and a patch size of 16, SAM's image encoder should output features
# with spatial dimensions 1024/16 = 64, and a channel dimension (e.g., 256).
segmentation_head = SAMDecoder(input_dim=256, output_channels=1)


# ------------------------------
# 6. Combine SAM with the Segmentation Head
# ------------------------------
class SAMFineTuner(nn.Module):
    def __init__(self, sam_model, segmentation_head):
        super().__init__()
        self.sam = sam_model
        self.segmentation_head = segmentation_head

    def forward(self, images):
        # Extract features using SAM's image encoder.
        features = self.sam.image_encoder(images)  # Expected shape: [batch, 256, 64, 64]
        # Pass the features through the custom segmentation head.
        masks = self.segmentation_head(features)  # Shape: [batch, 1, 64, 64]
        # Upsample to match the original image resolution (1024x1024)
        masks = F.interpolate(masks, size=images.shape[-2:], mode="bilinear", align_corners=False)
        return masks


model = SAMFineTuner(sam, segmentation_head)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------
# 7. Training Loop with Checkpoint Saving
# ------------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

num_epochs = 10
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)  # Expected output shape: [batch, 1, 1024, 1024]
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save checkpoint after each epoch
    save_path = f"/mnt/PhenomicsProjects/Detectron2/Apptainer/segmentanything_sandbox/home/ubuntu/model_checkpoints/finetuned_sam_epoch{epoch + 1}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Saved model checkpoint to {save_path}")
