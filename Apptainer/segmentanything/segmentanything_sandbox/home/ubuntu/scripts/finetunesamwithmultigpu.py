import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from segment_anything import sam_model_registry
import signal
import subprocess
import time

# Path to the system monitoring script
MONITOR_SCRIPT_PATH = "/home/appuser/scripts/SystemMonitor.sh"

def start_monitoring():
    print("Starting system monitoring...")
    process = subprocess.Popen(
        ["bash", MONITOR_SCRIPT_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid
    )
    time.sleep(2)
    retcode = process.poll()
    if retcode is not None:
        error_output = process.stderr.read().decode('utf-8').strip()
        print(f"Monitoring script failed to start (exit code: {retcode}).")
        print(f"stderr: {error_output}")
    else:
        print("Monitoring script started successfully, running in the background.")
    return process

def terminate_monitoring(process):
    print("Stopping system monitoring...")
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)

# Start monitoring
process = start_monitoring()

# --------------- PARAMETERS ---------------
target_size = (1024, 1024)
model_type = "vit_h"

# --------------- TRANSFORMS ---------------
image_transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor(),
])
mask_transform = transforms.Compose([
    transforms.Resize(target_size, interpolation=Image.NEAREST),
    transforms.ToTensor()
])

# --------------- DATASET CLASS ---------------
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, self.image_files[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx])).convert("L")
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return image, mask

# Dataset paths
images_dir = "/mnt/PhenomicsProjects/Detectron2/Datasets/breedingplots/output/images"
masks_dir = "/mnt/PhenomicsProjects/Detectron2/Datasets/breedingplots/output/masks"

dataset = SegmentationDataset(images_dir, masks_dir, image_transform, mask_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# --------------- LOAD ORIGINAL SAM ---------------
checkpoint_path = "/mnt/PhenomicsProjects/Detectron2/Apptainer/segmentanything/segmentanything_sandbox/home/ubuntu/model_checkpoints/sam_vit_h_4b8939.pth"
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

# Freeze encoder and prompt_encoder
for param in sam.image_encoder.parameters():
    param.requires_grad = False
for param in sam.prompt_encoder.parameters():
    param.requires_grad = False

# --------------- CUSTOM DECODER (mask_decoder) ---------------
class SAMDecoder(nn.Module):
    def __init__(self, input_dim=256, output_channels=1):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Replace the decoder
sam.mask_decoder = SAMDecoder(input_dim=256, output_channels=1)

# --------------- DEVICE SETUP ---------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam = nn.DataParallel(sam)
sam.to(device)

# --------------- TRAINING SETUP ---------------
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, sam.parameters()), lr=1e-4)
num_epochs = 1000

# --------------- TRAINING LOOP ---------------
sam.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        features = sam.module.image_encoder(images)
        outputs = sam.module.mask_decoder(features)
        outputs = F.interpolate(outputs, size=images.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Save checkpoint every 100 epochs
    if (epoch + 1) % 100 == 0:
        checkpoint_dir = "/mnt/PhenomicsProjects/Detectron2/Apptainer/segmentanything/segmentanything_sandbox/home/ubuntu/model_checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = f"{checkpoint_dir}/finetuned_sam_epoch{epoch+1}.pth"
        torch.save(sam.module.state_dict(), save_path)
        print(f"Saved checkpoint to {save_path}")

# Stop monitoring
terminate_monitoring(process)
