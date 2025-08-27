
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
    合成目标检测数据集，用于测试 YOLO 模型
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
        """创建合成的目标检测数据集"""
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']
        shapes = ['circle', 'rectangle', 'triangle']
        
        for i in range(self.num_samples):
            # 创建随机背景
            bg_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
            img = Image.new('RGB', self.img_size, bg_color)
            draw = ImageDraw.Draw(img)
            
            # 随机生成 1-3 个目标
            num_objects = random.randint(1, 3)
            objects = []
            
            for j in range(num_objects):
                # 随机选择形状和颜色
                shape = random.choice(shapes)
                color = random.choice(colors)
                
                # 随机位置和大小
                x1 = random.randint(50, self.img_size[0] - 150)
                y1 = random.randint(50, self.img_size[1] - 150)
                x2 = x1 + random.randint(80, 150)
                y2 = y1 + random.randint(80, 150)
                
                # 绘制形状
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
            
            # 保存图像
            img_path = os.path.join(self.root, f"synthetic_img_{i:03d}.jpg")
            img.save(img_path)
            
            # 创建标注
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
    将数据集转换为 YOLO 训练格式并保存
    """
    print("=== 转换为 YOLO 训练格式 ===")
    
    # 创建 YOLO 格式的目录结构
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # 类别映射
    class_names = ['circle', 'rectangle', 'triangle']
    class_mapping = {name: i for i, name in enumerate(class_names)}
    
    print(f"转换 {len(dataset)} 个样本...")
    
    # 转换每个样本
    for i, (img, annotation) in enumerate(dataset):
        # 保存图像
        img_filename = f"image_{i:04d}.jpg"
        img_path = os.path.join(images_dir, img_filename)
        img.save(img_path)
        
        # 创建 YOLO 格式的标签文件
        label_filename = f"image_{i:04d}.txt"
        label_path = os.path.join(labels_dir, label_filename)
        
        # 获取图像尺寸
        img_width, img_height = img.size
        
        # 转换标注为 YOLO 格式 (class x_center y_center width height)
        yolo_labels = []
        for obj in annotation['annotation']['object']:
            class_name = obj['name']
            if class_name in class_mapping:
                class_id = class_mapping[class_name]
                bbox = obj['bndbox']
                
                # 计算中心点和宽高（归一化）
                x_center = (float(bbox['xmin']) + float(bbox['xmax'])) / 2 / img_width
                y_center = (float(bbox['ymin']) + float(bbox['ymax'])) / 2 / img_height
                width = (float(bbox['xmax']) - float(bbox['xmin'])) / img_width
                height = (float(bbox['ymax']) - float(bbox['ymin'])) / img_height
                
                yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # 保存标签文件
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_labels))
    
    # 创建 YOLO 数据配置文件
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
    
    print(f"✅ 转换完成！")
    print(f"📁 图像目录: {images_dir}")
    print(f"📁 标签目录: {labels_dir}")
    print(f"📁 配置文件: {yaml_path}")
    print(f"📊 总样本数: {len(dataset)}")
    
    return yaml_path

def load_synthetic_dataset(root="./data"):
    """
    加载合成目标检测数据集
    """
    print("创建合成目标检测数据集...")
    ds = SyntheticDetectionDataset(root=root, num_samples=20)
    print(f"成功创建: 合成目标检测数据集, size={len(ds)}")
    return ds

def prepare_yolo_dataset(num_samples=20, output_dir="./data"):
    """
    准备完整的 YOLO 训练数据集
    """
    print("=== 准备 YOLO 训练数据集 ===")
    
    # 1. 创建合成数据集
    dataset = SyntheticDetectionDataset(root=output_dir, num_samples=num_samples)
    
    # 2. 转换为 YOLO 格式
    yaml_path = convert_to_yolo_format(dataset, output_dir)
    
    print(f"🎉 YOLO 数据集准备完成！")
    print(f"📁 配置文件路径: {yaml_path}")
    print(f"💡 现在可以使用 quick_train.py 进行训练了")
    
    return yaml_path

if __name__ == "__main__":
    # 准备 YOLO 训练数据集
    yaml_path = prepare_yolo_dataset(num_samples=100)
    print(f"\n=== 数据集信息 ===")
    print(f"配置文件: {yaml_path}")
    print(f"图像目录: ./data/images/")
    print(f"标签目录: ./data/labels/")
    print(f"类别: ['circle', 'rectangle', 'triangle']")
    print(f"样本数: 100")
