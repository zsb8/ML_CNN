
import os
import urllib.request
import zipfile
import json
import random
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection

class SyntheticDetectionDataset(Dataset):
    """
    Synthetic object detection dataset for testing YOLO models
    """
    def __init__(self, root="./data", num_samples=20, img_size=(640, 640)):
        self.root = root
        self.num_samples = num_samples
        self.img_size = img_size
        self.images = []
        self.annotations = []
        
        os.makedirs(root, exist_ok=True)
        self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self):
        """Create synthetic object detection dataset"""
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        shapes = ['circle', 'rectangle', 'triangle']
        
        for i in range(self.num_samples):
            # Create random background
            bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            img = Image.new('RGB', self.img_size, bg_color)
            draw = ImageDraw.Draw(img)
            
            # Randomly generate 1-3 objects
            num_objects = random.randint(1, 3)
            objects = []
            
            for j in range(num_objects):
                # Randomly select shape and color
                shape = random.choice(shapes)
                color = random.choice(colors)
                
                # Random position and size
                x1 = random.randint(50, self.img_size[0] - 150)
                y1 = random.randint(50, self.img_size[1] - 150)
                x2 = x1 + random.randint(80, 150)
                y2 = y1 + random.randint(80, 150)
                
                # Draw shape
                if shape == 'circle':
                    draw.ellipse([x1, y1, x2, y2], fill=color, outline='black', width=2)
                elif shape == 'rectangle':
                    draw.rectangle([x1, y1, x2, y2], fill=color, outline='black', width=2)
                elif shape == 'triangle':
                    points = [(x1, y2), ((x1 + x2) // 2, y1), (x2, y2)]
                    draw.polygon(points, fill=color, outline='black', width=2)
                
                objects.append({
                    "name": shape,
                    "bndbox": {
                        "xmin": x1,
                        "ymin": y1,
                        "xmax": x2,
                        "ymax": y2
                    }
                })
            
            # Save image
            img_path = os.path.join(self.root, f"synthetic_img_{i:03d}.jpg")
            img.save(img_path)
            
            # Create annotation
            annotation = {
                "annotation": {
                    "filename": f"synthetic_img_{i:03d}.jpg",
                    "size": {"width": self.img_size[0], "height": self.img_size[1]},
                    "object": objects
                }
            }
            
            self.images.append(img_path)
            self.annotations.append(annotation)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        annotation = self.annotations[idx]
        
        img = Image.open(img_path).convert('RGB')
        return img, annotation

def convert_to_yolo_format(dataset, output_dir="./data"):
    """
    Convert dataset to YOLO training format and save
    """
    print("=== Converting to YOLO training format ===")
    
    # Create YOLO format directory structure
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Class mapping
    class_names = ['circle', 'rectangle', 'triangle']
    class_mapping = {name: i for i, name in enumerate(class_names)}
    
    print(f"Converting {len(dataset)} samples...")
    
    # Convert each sample
    for i, (img, annotation) in enumerate(dataset):
        # Save image
        img_filename = f"image_{i:04d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)
        
        # Create YOLO format label file
        label_filename = f"image_{i:04d}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        # Get image dimensions
        img_width, img_height = img.size
        
        # Convert annotations to YOLO format (class x_center y_center width height)
        yolo_labels = []
        for obj in annotation['annotation']['object']:
            class_name = obj['name']
            if class_name in class_mapping:
                class_id = class_mapping[class_name]
                bbox = obj['bndbox']
                
                # Calculate center point and width/height (normalized)
                x_center = (float(bbox['xmin']) + float(bbox['xmax'])) / 2 / img_width
                y_center = (float(bbox['ymin']) + float(bbox['ymax'])) / 2 / img_height
                width = (float(bbox['xmax']) - float(bbox['xmin'])) / img_width
                height = (float(bbox['ymax']) - float(bbox['ymin'])) / img_height
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Save label file
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_labels))
    
    # Create YOLO data configuration file
    yaml_content = f"""# YOLO dataset config
        path: {os.path.abspath(output_dir)}
        train: images
        val: images

        # Classes
        nc: {len(class_names)}
        names: {class_names}
        """
    
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"✅ Conversion completed!")
    print(f"📁 Images directory: {images_dir}")
    print(f"📁 Labels directory: {labels_dir}")
    print(f"📁 Config file: {yaml_path}")
    print(f"📊 Total samples: {len(dataset)}")
    
    return yaml_path

def load_synthetic_dataset(root="./data"):
    """
    Load synthetic object detection dataset
    """
    print("Creating synthetic object detection dataset...")
    ds = SyntheticDetectionDataset(root=root, num_samples=20)
    print(f"Successfully created: synthetic object detection dataset, size={len(ds)}")
    return ds

def prepare_yolo_dataset(num_samples=20, output_dir="./data"):
    """
    Prepare complete YOLO training dataset
    """
    print("=== Preparing YOLO training dataset ===")
    
    # 1. Create synthetic dataset
    dataset = SyntheticDetectionDataset(root=output_dir, num_samples=num_samples)
    
    # 2. Convert to YOLO format
    yaml_path = convert_to_yolo_format(dataset, output_dir)
    
    print(f"🎉 YOLO dataset preparation completed!")
    print(f"📁 Config file path: {yaml_path}")
    print(f"💡 Now you can use quick_train.py for training")
    
    return yaml_path

if __name__ == "__main__":
    # Prepare YOLO training dataset
    yaml_path = prepare_yolo_dataset(num_samples=100)
    print(f"\n=== Dataset Information ===")
    print(f"Config file: {yaml_path}")
    print(f"Images directory: ./data/images/")
    print(f"Labels directory: ./data/labels/")
    print(f"Classes: ['circle', 'rectangle', 'triangle']")
    print(f"Sample count: 100")
