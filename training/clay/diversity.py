import os
import glob
import lpips
import numpy as np
import torch
from PIL import Image
import random

def load_images(directory):
    image_paths = glob.glob(os.path.join(directory, '*.png'))
    images = [Image.open(image_path) for image_path in image_paths]
    images_sample = random.sample(images,2)
    return images_sample

def preprocess_images(images):
    processed_images = [np.array(image) for image in images]
    processed_images = [(image.astype(np.float32) / 255.0) * 2.0 - 1.0 for image in processed_images]
    return processed_images

def calculate_lpips_distances(processed_images):
    lpips_net = lpips.LPIPS(net='vgg')
    num_images = len(processed_images)
    lpips_scores = []

    for i in range(num_images):
        for j in range(i + 1, num_images):
            print(i, " ", j, " ", num_images)
            dist = lpips_net.forward(
                torch.from_numpy(processed_images[i]).unsqueeze(0).permute(0, 3, 1, 2),  # Adjust the tensor shape
                torch.from_numpy(processed_images[j]).unsqueeze(0).permute(0, 3, 1, 2)   # Adjust the tensor shape
            ).item()
            lpips_scores.append(dist)

    return np.array(lpips_scores)

if __name__ == '__main__':
    images_directory = './exp_metrics2'

    images = load_images(images_directory)
    processed_images = preprocess_images(images)
    lpips_scores = calculate_lpips_distances(processed_images)

    diversity_score = np.mean(lpips_scores)  # You can also use np.max(lpips_scores)
    print("Diversity Score:", diversity_score)


