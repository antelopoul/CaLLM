import json
import os
import torch
from PIL import Image
import numpy as np

# Path to the folder containing the images
image_folder = "cc_sbu_align/cc_sbu/cc_sbu_align/image"

# Load data from JSON file
with open("cc_sbu_align/cc_sbu/cc_sbu_align/filter_cap.json", "r") as file:
    data = json.load(file)

# Function to load image tensor
def load_image(image_path):
    with Image.open(image_path) as img:
        img_np = np.array(img)  # Convert image to NumPy array
        return torch.tensor(img_np)  # Create tensor from NumPy array

# Iterate over annotations
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    caption = annotation["caption"]

    # Print image ID and caption
    print(f"Image ID: {image_id}")
    print(f"Caption: {caption}")

    caption_size = len(caption)  # or len(caption.split()) for word count
    print(f"Caption size: {caption_size}")

    # Check if image file exists
    image_path = os.path.join(image_folder, f"{image_id}.jpg")
    if os.path.exists(image_path):
        # Load and print image shape
        image_tensor = load_image(image_path)
        print(f"Image shape: {image_tensor.shape}")
    else:
        print(f"Image file not found for image ID: {image_id}")

    print()  # Add a newline for readability
