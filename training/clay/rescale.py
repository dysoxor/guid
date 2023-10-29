import os
from PIL import Image

input_folder = './out_test_lt_croped'
output_folder = './out_test_lt_croped_resized'
target_size = (256, 256)

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of image files in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

for image_file in image_files:
    input_path = os.path.join(input_folder, image_file)
    output_path = os.path.join(output_folder, image_file.split(".")[0]+".png")
    
    # Open the image using PIL
    img = Image.open(input_path)
    
    # Resize the image
    resized_img = img.resize(target_size)
    
    # Save the resized image
    resized_img.save(output_path)
    
    #print(f"Resized: {input_path} -> {output_path}")

print("Resizing complete.")

