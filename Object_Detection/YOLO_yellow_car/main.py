import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 加载 YOLO 模型
yolo_model = YOLO('yolov8n.pt')  # 使用 YOLOv8 模型
# 加载 CLIP 模型
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def analyze_color(cropped_img, target_color="yellow"):
    """分析裁剪图像的主要颜色"""
    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # 定义颜色范围 - 添加更多颜色支持
    color_ranges = {
        "yellow": ([15, 50, 50], [35, 255, 255]),  # 更宽松的黄色范围
        "red": ([0, 100, 100], [10, 255, 255]),  # 红色在HSV中是0度
        "blue": ([100, 100, 100], [130, 255, 255]),
        "green": ([35, 50, 50], [85, 255, 255]),  # 添加绿色范围
        "white": ([0, 0, 200], [180, 30, 255]),
        "black": ([0, 0, 0], [180, 255, 100])  # 更宽松的黑色范围：明度上限100
    }
    
    if target_color not in color_ranges:
        return 0.0
    
    lower, upper = color_ranges[target_color]
    lower = np.array(lower)
    upper = np.array(upper)
    
    # 创建掩码
    mask = cv2.inRange(hsv, lower, upper)
    
    # 计算目标颜色像素占比
    total_pixels = mask.shape[0] * mask.shape[1]
    color_pixels = cv2.countNonZero(mask)
    color_ratio = color_pixels / total_pixels
    
    return color_ratio

# 自然语言目标检测函数
def detect_objects_with_text(image, text_query, confidence_threshold=0.5, similarity_threshold=0.2, color_threshold=0.05, num=1):
    # 使用 YOLO 进行目标检测
    detections = yolo_model(image)
    boxes = detections[0].boxes.xyxy.cpu().numpy()  # 获取候选框坐标
    scores = detections[0].boxes.conf.cpu().numpy()  # 获取置信度
    print(f"YOLO检测到 {len(boxes)} 个目标")
    # 过滤低置信度框
    valid_boxes = []
    for i, score in enumerate(scores):
        if score > confidence_threshold:
            x1, y1, x2, y2 = map(int, boxes[i])
            cropped_img = image[y1:y2, x1:x2]  # 截取目标框区域
            valid_boxes.append((cropped_img, (x1, y1, x2, y2)))
    print(f"置信度过滤后剩余 {len(valid_boxes)} 个目标")
    
    # 使用 CLIP 匹配文本和候选框
    matched_boxes = []
    all_similarities = []  # 记录所有相似度用于分析
    
    for idx, (cropped_img, box_coords) in enumerate(valid_boxes):
        # 将目标框区域转换为 CLIP 可处理的格式
        clip_input = clip_processor(images=[cropped_img], return_tensors="pt").to("cpu")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**clip_input)
            text_input = clip_processor(text=[text_query], return_tensors="pt").to("cpu")
            text_features = clip_model.get_text_features(**text_input)
        # 计算图像和文本特征的相似度
        similarity = torch.cosine_similarity(image_features, text_features).item()
        all_similarities.append(similarity)
        # 颜色分析
        color_ratio = 0.0
        if "yellow" in text_query.lower():
            color_ratio = analyze_color(cropped_img, "yellow")
        elif "red" in text_query.lower():
            color_ratio = analyze_color(cropped_img, "red")
        elif "blue" in text_query.lower():
            color_ratio = analyze_color(cropped_img, "blue")
        elif "green" in text_query.lower():
            color_ratio = analyze_color(cropped_img, "green")
        elif "white" in text_query.lower():
            color_ratio = analyze_color(cropped_img, "white")
        elif "black" in text_query.lower():
            color_ratio = analyze_color(cropped_img, "black")
        print(f"目标 {idx+1}: CLIP相似度 = {similarity:.3f}, 颜色占比 = {color_ratio:.3f}")
        # 综合判断：CLIP相似度和颜色分析
        if similarity > similarity_threshold and color_ratio > color_threshold:
            matched_boxes.append((box_coords, similarity, color_ratio))

    # 分析相似度分布
    if all_similarities:
        max_sim = max(all_similarities)
        min_sim = min(all_similarities)
        avg_sim = sum(all_similarities) / len(all_similarities)
        print(f"\n相似度统计:")
        print(f"  最高相似度: {max_sim:.3f}")
        print(f"  最低相似度: {min_sim:.3f}")
        print(f"  平均相似度: {avg_sim:.3f}")
        print(f"  当前阈值: {similarity_threshold:.3f}")
    # 按颜色占比排序（对于颜色查询）
    if "yellow" in text_query.lower() or "red" in text_query.lower() or "blue" in text_query.lower() or "green" in text_query.lower() or "white" in text_query.lower() or "black" in text_query.lower():
        matched_boxes = sorted(matched_boxes, key=lambda x: x[2], reverse=True)  # 按颜色占比排序
    else:
        matched_boxes = sorted(matched_boxes, key=lambda x: x[1], reverse=True)  # 按相似度排序
    if not matched_boxes:
        print(f"\n未找到匹配的目标")
        print("建议：可以尝试降低阈值或使用更通用的描述词")
        return None
    print(f"\n找到 {len(matched_boxes)} 个匹配的目标")

    # 绘制检测结果
    for idx, (box, similarity, color_ratio) in enumerate(matched_boxes[:num]):
        x1, y1, x2, y2 = box
        # 绘制检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 创建标签文本
        label_text = f"No.{idx+1}: {color_ratio:.1%}"
        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # 绘制标签背景
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        # 绘制标签文本
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image

if __name__ == '__main__':
    # 主程序
    image_path = "pictures/cars.png"  
    text_query = "yellow car"
    print(f"正在加载图片: {image_path}")
    print(f"搜索目标: {text_query}")
    # 加载图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
    else:
        print(f"图片加载成功，尺寸: {image.shape}")
        # 进行目标检测 - 使用更宽松的参数
        result_image = detect_objects_with_text(
            image, 
            text_query, 
            similarity_threshold=0.2,   # 降低CLIP相似度阈值
            color_threshold=0.05,       # 降低颜色占比阈值（5%以上黄色像素）
            num=3                       # 只显示最blue的1辆车
        )
        if result_image is not None:
            print("检测完成，正在显示结果...")
            # 显示最终结果
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Result: {text_query}")
            plt.axis("off")
            plt.show()
        else:
            print("没有检测到匹配的目标")