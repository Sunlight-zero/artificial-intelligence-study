from typing import Optional, Callable
import os

import json
import torch
from diffusers import DDPMPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


lr = 0.001
batch_size = 32
device = torch.device("cuda")
num_epochs = 1000
train_timesteps = 100

cache_dir = "/home/sunlight/learning/ai/generative_models"

dataset_path = "/home/sunlight/learning/diffusers/dataset/text2image_lr_dataset"
meta_path = "/home/sunlight/learning/diffusers/dataset/image_list.txt"


clip_model_name = "openai/clip-vit-base-patch32"
tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
text_encoder = CLIPTextModel.from_pretrained(clip_model_name)

class TextToImageDataset(Dataset):
    def __init__(self, dir_path: str, meta_file: str, transforms: Optional[Callable]=None):
        super().__init__()
        with open(meta_file) as f:
            files = f.read().split('\n')
        self.dataset = []
        self.transforms = transforms
        
        cnt = 0
        for filename in tqdm(files, desc="Image Loading"):
            image = Image.open(os.path.join(dir_path, filename + ".jpg"))
            if transforms:
                image = transforms(image)
            with open(os.path.join(dir_path, filename + ".json")) as json_file:
                prompt_json = json.load(json_file)
            prompt = prompt_json["prompt"]
            tokenized_prompt = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids
            with torch.no_grad():
                embedded_prompt = text_encoder(
                    tokenized_prompt
                ).last_hidden_state.squeeze(0)
            self.dataset.append({"image": image, 
                                 "prompt": prompt,
                                 "embedded": embedded_prompt
                                 })
            cnt += 1
        
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

dataset = TextToImageDataset(dataset_path, meta_path, ToTensor())
dataloader = DataLoader(dataset, batch_size, shuffle=True)

unet = UNet2DConditionModel(
    sample_size=64, # 输入图像（或隐表示张量）的大小，这里为 64x64
    in_channels=3, # 输入通道数
    out_channels=3, # 输出通道数
    layers_per_block=2, # 每层的
    block_out_channels=(32, 64, 128, 256), # 输出通道数
    cross_attention_dim=512, # 图像嵌入 / 文本嵌入特征数
    down_block_types=(
        "DownBlock2D", # 普通下采样层，卷积
        "CrossAttnDownBlock2D",  # 交叉注意力层
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "UpBlock2D",
    )
)

scheduler = DDPMScheduler(num_train_timesteps=train_timesteps)
optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

unet.to(device)

for epoch in tqdm(range(num_epochs)):
    for batch in tqdm(dataloader):
        images = batch["image"]
        prompts = batch["prompt"]
        encoder_hidden_states = batch["embedded"]
        
        this_batch_size = images.shape[0]

        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, train_timesteps, (this_batch_size,)
        ).long()
        noised_images = scheduler.add_noise(images, noise, timesteps)

        # tokenized_prompts = tokenizer(
        #     prompts,
        #     padding="max_length",
        #     max_length=tokenizer.model_max_length,
        #     truncation=True,
        #     return_tensors="pt"
        # ).input_ids
        # with torch.no_grad():
        #     encoder_hidden_states = text_encoder(
        #         tokenized_prompts
        #     ).last_hidden_state
        
        noised_images = noised_images.to(device)
        timesteps = timesteps.to(device)
        encoder_hidden_states = encoder_hidden_states.to(device)
        noise = noise.to(device)

        noise_pred = unet(noised_images, timesteps, encoder_hidden_states).sample

        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        loss.backward()
        optimizer.step()
    
    if epoch % 100 == 99:
        pipeline = DDPMPipeline(
            unet=unet,
            scheduler=scheduler,
        )
        pipeline.save_pretrained(f"ckpt/ddpm-checkpoint-{epoch}")

# pipeline = DDPMPipeline(
#     unet=unet, scheduler=scheduler,
# )
# pipeline.save_pretrained(f"ckpt/ddpm-checkpoint-{epoch}")
