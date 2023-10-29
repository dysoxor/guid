import os
from PIL import Image

source_directory = "./out_test_t"
destination_directory = "./out_test_t_croped"

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

crop_left = 112
crop_right = 112
crop_top = 0
crop_bottom = 0

# Iterate through all images in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(source_directory, filename)
        image = Image.open(image_path)
        
        # Crop the image
        cropped_image = image.crop((crop_left, crop_top, 512 - crop_right, 512 - crop_bottom))
        
        # Save the cropped image to the destination directory
        cropped_image.save(os.path.join(destination_directory, filename))

print("Cropping and saving of images completed.")

