import torch
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from tqdm import tqdm


image_size = 64
model_path = "/home/sunlight/learning/ai/generative_models/diffusion/ckpt/ddpm-checkpoint-99"
device = torch.device("cuda")
prompt = ""
steps = 1000
sample_batch = 1


clip_model_name = "openai/clip-vit-base-patch32"
tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
text_encoder.eval()

text_inputs = tokenizer(
    [prompt],
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
)
input_ids = text_inputs.input_ids
condition = text_encoder(input_ids).last_hidden_state.to(device)

pipeline = DDPMPipeline.from_pretrained(model_path)
unet: UNet2DConditionModel = pipeline.unet.to(device)
scheduler: DDPMScheduler = pipeline.scheduler
unet.eval()

noised_image = torch.randn((sample_batch, 3, image_size, image_size)).to(device)

for t in tqdm(scheduler.timesteps):
    timestep = torch.tensor([t], device=device)
    noise_pred = unet(noised_image, timestep, condition).sample
    noised_image = scheduler.step(noise_pred, t, noised_image).prev_sample

image = (noised_image.clamp(0, 1).cpu().permute(0, 2, 3, 1) * 255).byte().numpy()[0]
Image.fromarray(image).save("generated_image.png")

