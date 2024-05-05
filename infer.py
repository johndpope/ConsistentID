import torch
import os
from diffusers.utils import load_image
from diffusers import EulerDiscreteScheduler
from pipline_StableDiffusion_ConsistentID import ConsistentIDStableDiffusionPipeline
import sys


# TODO import base SD model and pretrained ConsistentID model
device = "cuda"
base_model_path = "SG161222/Realistic_Vision_V6.0_B1_noVAE"
consistentID_path = "./ConsistentID_model_facemask_pretrain_50w.bin" # pretrained ConsistentID model


# negative_embedding_path = "UnrealisticDream.pt"
# negative_embedding_path = "BadDream.pt"
# negative_prompt_embeds = torch.load(negative_embedding_path)

# Gets the absolute path of the current script
script_directory = os.path.dirname(os.path.realpath(__file__))

### Load base model
pipe = ConsistentIDStableDiffusionPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    use_safetensors=False
).to(device)

### Load consistentID_model checkpoint
pipe.load_ConsistentID_model(
    os.path.dirname(consistentID_path),
    subfolder="",
    weight_name=os.path.basename(consistentID_path),
    trigger_word="img",
)     

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# lora_model_name = os.path.basename(lora_path)
# pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_model_name) # trigger: HTA
### If there's a specific adapter name defined for this LoRA, use it; otherwise, the default might work.
### Ensure 'adapter_name' matches what you intend to use or remove if not needed in your setup.
# pipe.set_adapter_settings(adapter_name="licking_my_dick.sd.v1.2.safetensors") # Uncomment and adjust as necessary
# pipe.set_adapters(,["ConsistentID", "more_art-full"] adapter_weights=[1.0, 0.5]) # TODO
### Fuse the loaded LoRA into the pipeline
# pipe.fuse_lora()
# pipe.load_lora_weights(".", weight_name="orgasm_face_v10.safetensors")
# pipe.load_lora_weights(".", weight_name="licking_my_dick.sd.v1.2.safetensors")
# pipe.set_adapters(
#     ["orgasm_face_v10", "licking_my_dick.sd.v1.2"],
#     adapter_weights=[0.8, 1]
# )

### input image TODO
# select_images = load_image(script_directory+"/images/M.jpg")
select_images = load_image('/home/oem/Pictures/Screenshots/old/4.png') 
# hyper-parameter
num_steps = 50
merge_steps = 20
# Prompt
# prompt = "DDlipbit_v2:0.5 img"
prompt = "eyes closed, mouth open, orgasm,sexy asian, realistic, solo, 1gril,  orgasm_face, orgasm_face_v10:0.8, photorealistic, , nsfw, bed"
negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry"

#Extend Prompt
# prompt =  prompt + ", natural shading, 85mm, f/1.4, ISO 200, 1/160s:0.75), perfect anatomy"
negtive_prompt_group="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, UnrealisticDream"
negative_prompt = negative_prompt + negtive_prompt_group

generator = torch.Generator(device=device) #.manual_seed(2024)
pipe.safety_checker = lambda images, clip_input: (images, False)


# cross_attention_kwargs = {
#     "scale": 0.8,  # Adjust the scale value to control the LoRA strength
# }

# Generate the images
    # negative_prompt=negative_prompt,
result = pipe(
    prompt=prompt,
    width=512,    
    height=768,
    input_id_images=select_images,
    # negative_prompt_embeds=negative_prompt_embeds,
    negative_prompt=negative_prompt,
    num_images_per_prompt=1,
    num_inference_steps=num_steps,
    start_merge_step=merge_steps,
    generator=torch.manual_seed(0),
    # cross_attention_kwargs=cross_attention_kwargs
)

# Ensure the directory exists

if not os.path.exists(script_directory + "/images"):
    os.makedirs(script_directory + "/images")

# Save each image with a unique filename
for i, image in enumerate(result.images):
    image.save(f"{script_directory}/images/sample_{i+1}.jpg")

