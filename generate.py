import os
from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.uniformer import UniformerDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

import json

torch.cuda.empty_cache()
def force_cudnn_initialization():
	s = 32
	dev = torch.device('cuda')
	torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
force_cudnn_initialization()
apply_uniformer = UniformerDetector()
# Rest of the code remains the same
model = create_model('./models/cldm_v21.yaml').cpu()
#model.load_state_dict(load_state_dict('./lightning_logs/version_4/checkpoints/epoch=7-step=95255.ckpt', location='cuda'))
#model.load_state_dict(load_state_dict('./lightning_logs/version_5/checkpoints/epoch=9-step=89489.ckpt', location='cuda'))
model.load_state_dict(load_state_dict('./lightning_logs/version_6/checkpoints/epoch=10-step=98438.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

# Define the input and output folders
#input_folder = './training/clay/test2'
input_folder = './training/clay/test3'
output_folder = './training/clay/out_test_t'

# List all image files in the input folder
image_files1 = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png'))]
image_files = random.sample(image_files1, 360)

def generate_images_for_folder():
	a_prompt = 'best quality, extremely detailed'
	n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
	eta = 0.0
	
	scale = 15
	ddim_steps = 100
	detect_resolution = 512
	guess_mode = False
	strength = 1.0
	image_resolution = 512
	num_samples = 1
	prompt = 'High quality, detailed, and professional app interface'
	is_default = False
	content = None
	json_entries = None
	if not is_default:
		with open('./training/clay/prompt_test.json', 'r') as file:
			content = file.read()
		json_entries = content.split('\n')
	for image_file in image_files:
		for it in range(1):
			if not is_default:
				for entry in json_entries:
					try:
						parsed_entry = json.loads(entry)
						if parsed_entry["source"].split("/")[-1] == image_file:
							prompt = parsed_entry["prompt"]
							break
					except json.JSONDecodeError:
						pass
				if prompt == 'High quality, detailed, and professional app interface':
					print('error')
			seed = -1
			image_path = os.path.join(input_folder, image_file)
			#input_image = cv2.imread(image_path)  # Load input image
			input_image = cv2.imread('../downloads/colored_image.png')
			if input_image is None:
				continue
		    
		    # Rest of the processing code remains the same
			with torch.no_grad():
				input_image = HWC3(input_image)
				detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
				img = resize_image(input_image, image_resolution)
				H, W, C = img.shape

				detected_map = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_NEAREST)

				control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
				control = torch.stack([control for _ in range(num_samples)], dim=0)
				control = einops.rearrange(control, 'b h w c -> b c h w').clone()

				if seed == -1:
					seed = random.randint(0, 65535)
				seed_everything(seed)

				if config.save_memory:
					model.low_vram_shift(is_diffusing=False)

				cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
				un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
				shape = (4, H // 8, W // 8)

				if config.save_memory:
					model.low_vram_shift(is_diffusing=True)

				model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
				samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
								                             shape, cond, verbose=False, eta=eta,
								                             unconditional_guidance_scale=scale,
								                             unconditional_conditioning=un_cond)

				if config.save_memory:
					model.low_vram_shift(is_diffusing=False)

				x_samples = model.decode_first_stage(samples)
				x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

				results = [x_samples[i] for i in range(num_samples)]

		    # Save the generated images to the output folder
			for idx, result_image in enumerate(results):
				output_image_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_{it}.png")
				cv2.imwrite(output_image_path, result_image)

generate_images_for_folder()

