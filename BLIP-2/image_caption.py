import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from PIL import Image
import argparse
import requests
import json
import cv2
import os
from glob import glob

def load_demo_image(img_url,image_size, device):
    raw_image = cv2.imread(img_url)
    if raw_image is None:
        raise ValueError(f"Image at path '{img_url}' could not be loaded.")
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    
    # Convert from NumPy array to PIL Image
    raw_image = Image.fromarray(raw_image).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    
    image = transform(raw_image).to(device)
    image = image.unsqueeze(0)
    return raw_image

def load_images_from_folder(folder_path, image_size, device):
  """Loads all images from a folder and applies preprocessing."""
  image_paths = glob(os.path.join(folder_path, "*.jpg")) + glob(os.path.join(folder_path, "*.png"))
  images = []
  for image_path in image_paths:
    try:
      raw_image = load_demo_image(image_path, image_size, device)
      # raw_image = cv2.imread(image_path)
      # if raw_image is None:
      #   raise ValueError(f"Image at path '{image_path}' could not be loaded.")
      # raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

      # Convert from NumPy array to PIL Image
      # raw_image = Image.fromarray(raw_image).convert('RGB')

      # transform = transforms.Compose([
      #   transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
      #   transforms.ToTensor(),
      #   transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
      # ])

      # image = transform(raw_image).to(device)
      # image = image.unsqueeze(0)

      images.append((image_path, raw_image))
    except Exception as e:
      print(f"Error loading image {image_path}: {e}")
  return images

def main():
   parser = argparse.ArgumentParser(description='Generate captions for images in a folder')
   parser.add_argument('--folder_path', type=str, help='Path to the folder containing images')
   parser.add_argument('--output_path', type=str, help='Path to save the output JSON file')
   args = parser.parse_args()

   device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
   print(f"Using device: {device}")
   model, vis_processors, _ = load_model_and_preprocess(
   name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device)
   vis_processors.keys()
   captions = []
   folder_path = args.folder_path
   raw_images = load_images_from_folder(folder_path, 256, 'cuda') 
   for image_path, raw_image in raw_images:
      with torch.no_grad():
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption_text = model.generate({"image": image})
        image_id = os.path.splitext(os.path.basename(image_path))[0]  # Extract image ID from image path
        captions.append({"image_id": image_id, "caption": caption_text})

    # Write captions to a JSON file
   with open(args.output_path, 'w') as f:
      json.dump(captions, f)

if __name__ == '__main__':
    main()