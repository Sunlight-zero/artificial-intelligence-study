import torch
from diffusers import DDPMPipeline, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"


model_path = "/home/sunlight/learning/diffusers/diffusion_from_scratch/ckpt/ddpm-checkpoint-9"
prompt = "a photo of a cat"  # 可替换为任意提示词

# 加载训练好的模型和调度器
pipeline = DDPMPipeline.from_pretrained(model_path)
model = pipeline.unet.to(device)
model.eval()
scheduler: DDPMScheduler = pipeline.scheduler

# 加载CLIP模型处理文本条件
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_encoder.eval()

# 输入文本提示
text_inputs = tokenizer(
    [prompt],  # 保持batch维度
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)
input_ids = text_inputs.input_ids.to(device)

# 生成文本条件特征
with torch.no_grad():
    encoder_hidden_states = text_encoder(input_ids).last_hidden_state

# 生成初始噪声
batch_size = 1
num_channels = 3
image_size = 64
noise = torch.randn((batch_size, num_channels, image_size, image_size), device=device)

# 设置去噪步数
scheduler.set_timesteps(num_inference_steps=1000)

# 逐步去噪
for t in tqdm(scheduler.timesteps, desc="Generate"):
    timestep = torch.tensor([t], device=device)
    
    # 预测噪声残差
    with torch.no_grad():
        noise_pred = model(noise, timestep, encoder_hidden_states).sample
    
    # 更新图像
    noised_image = scheduler.step(noise_pred, t, noised_image).prev_sample

# 后处理并保存图像
image = (noised_image.clamp(0, 1).cpu().permute(0, 2, 3, 1) * 255).byte().numpy()[0]
Image.fromarray(image).save("generated_image.png")
