import torch
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DModel
import torchvision as tv
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl


# Hyperparameters
learning_rate = 1e-4
total_timesteps = 1000
device = torch.device("cuda")
batch_size = 128
num_epochs = 100


# Step 1. Load the butterfly dataset
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
image_size = 32
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

dataset.set_transform(transform)
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Step 2. Define denoised U-Net
unet = UNet2DModel(
    sample_size=image_size,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 128, 256),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "UpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
    )
)

# Step 3. Scheduler
scheduler = DDPMScheduler(
    total_timesteps, 
    beta_start=0.0001, beta_end=0.02, beta_schedule="linear"
)

# Step 4. Training
class DiffusionTrainPl(pl.LightningModule):
    def __init__(self, unet: nn.Module, scheduler: DDPMScheduler):
        super().__init__()
        self.unet = unet
        self.scheduler = scheduler
    
    def training_step(self, batch, batch_idx):
        images = batch["images"]
        batch_size = images.shape[0]
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, total_timesteps,
            (batch_size,),
            device=self.device
        ).long()
        noised_images = self.scheduler.add_noise(images, noise, timesteps)
        noise_pred = unet(noised_images, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.unet.parameters(), lr=learning_rate)
    
    def on_train_epoch_end(self):
        # 每5个epoch打印一次平均损失
        if (self.current_epoch + 1) % 5 == 0:
            avg_loss = self.trainer.callback_metrics["train_loss"]
            print(f"Epoch: {self.current_epoch+1}, Loss: {avg_loss:.4f}")

pl_model = DiffusionTrainPl(unet, scheduler)
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator="auto",
    devices="auto"
)
trainer.fit(pl_model, train_dataloader)

# Step 5. Save pipeline
my_pipe = DDPMPipeline(unet=unet, scheduler=scheduler)
my_pipe.save_pretrained(f"ckpt/butterfly")

# Step 6. Generate
my_pipe = my_pipe.to(device)
pipeline_output = my_pipe(batch_size=10)
generated_image = pipeline_output.images
generated_image

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.subplots_adjust(wspace=0.1, hspace=0.1) 

for i, ax in enumerate(axes.flat):
    ax.imshow(generated_image[i])
    ax.axis('off')

plt.show()
