import os
from PIL import Image, ImageOps

input_directory = "./test3"  # Replace with the path to your input directory
output_directory = "./test3"  # You can also use a different output directory

desired_width = 512
desired_height = 512

def process_images():
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(input_directory, image_file)
        
        # Open the original image
        original_image = Image.open(image_path)

        # Add borders to transform it to the desired width
        original_width, original_height = original_image.size
        border_width = (desired_width - original_width) // 2

        # Create a new image with white borders
        new_image = Image.new("RGB", (desired_width, original_height), (200, 200, 200))

        # Paste the original image onto the new image with borders
        new_image.paste(original_image, (border_width, 0))

        # Resize the bordered image to the desired size
        resized_image = new_image.resize((desired_width, desired_height), Image.ANTIALIAS)

        # Save the processed image
        output_image_path = os.path.join(output_directory, image_file)
        resized_image.save(output_image_path)

    print("Image processing complete.")

process_images()

