import os
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from PIL import Image

# Scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    return asarray(images_list)

# Calculate Frechet Inception Distance
def calculate_fid(model, images1, images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Prepare the Inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# Path to the folders containing generated and ground-truth GUI designs
generated_folder = './exp_metrics3/1'
#generated_folder = './ganspiration/out'
ground_truth_folder = './exp_metrics3/2'

# Load and preprocess images from the folders
def load_and_preprocess_images(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        image = image.resize((299, 299))
        image = np.array(image)
        images.append(image)

    return preprocess_input(np.array(images))

generated_images = load_and_preprocess_images(generated_folder)
ground_truth_images = load_and_preprocess_images(ground_truth_folder)

# Calculate FID between generated and ground-truth images
fid = calculate_fid(model, generated_images, ground_truth_images)
print(f'FID between generated and ground-truth GUI designs: {fid:.3f}')

