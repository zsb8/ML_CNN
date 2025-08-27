import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load YOLO model
yolo_model = YOLO('yolov8n.pt')  # Use YOLOv8 model
# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def analyze_color(cropped_img, target_color="yellow"):
    """Analyze the main color of the cropped image"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges - add support for more colors
    color_ranges = {
        "yellow": ([15, 50, 50], [35, 255, 255]),  # More relaxed yellow range
        "red": ([0, 100, 100], [10, 255, 255]),  # Red is 0 degrees in HSV
        "blue": ([100, 100, 100], [130, 255, 255]),
        "green": ([35, 50, 50], [85, 255, 255]),  # Add green range
        "white": ([0, 0, 200], [180, 30, 255]),
        "black": ([0, 0, 0], [180, 255, 100])  # More relaxed black range: brightness upper limit 100
    }
    
    if target_color not in color_ranges:
        return 0.0
    
    lower, upper = color_ranges[target_color]
    lower = np.array(lower)
    upper = np.array(upper)
    
    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    
    # Calculate target color pixel ratio
    total_pixels = mask.shape[0] * mask.shape[1]
    color_pixels = cv2.countNonZero(mask)
    color_ratio = color_pixels / total_pixels
    
    return color_ratio

# Natural language object detection function
def detect_objects_with_text(image, text_query, confidence_threshold=0.5, similarity_threshold=0.2, color_threshold=0.05, num=1):
    # Use YOLO for object detection
    detections = yolo_model(image)
    boxes = detections[0].boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
    scores = detections[0].boxes.conf.cpu().numpy()  # Get confidence scores
    print(f"YOLO detected {len(boxes)} objects")
    # Filter low confidence boxes
    valid_boxes = []
    for i, score in enumerate(scores):
        if score > confidence_threshold:
            x1, y1, x2, y2 = map(int, boxes[i])
            cropped_img = image[y1:y2, x1:x2]  # Crop target box region
            valid_boxes.append((cropped_img, (x1, y1, x2, y2)))
    print(f"After confidence filtering, {len(valid_boxes)} objects remain")
    
    # Use CLIP to match text and candidate boxes
    matched_boxes = []
    all_similarities = []  # Record all similarities for analysis
    
    for idx, (cropped_img, box_coords) in enumerate(valid_boxes):
        # Convert target box region to CLIP processable format
        clip_input = clip_processor(images=[cropped_img], return_tensors="pt").to("cpu")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**clip_input)
            text_input = clip_processor(text=[text_query], return_tensors="pt").to("cpu")
            text_features = clip_model.get_text_features(**text_input)
        # Calculate similarity between image and text features
        similarity = torch.cosine_similarity(image_features, text_features).item()
        all_similarities.append(similarity)
        # Color analysis
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
        print(f"Object {idx+1}: CLIP similarity = {similarity:.3f}, color ratio = {color_ratio:.3f}")
        # Comprehensive judgment: CLIP similarity and color analysis
        if similarity > similarity_threshold and color_ratio > color_threshold:
            matched_boxes.append((box_coords, similarity, color_ratio))

    # Analyze similarity distribution
    if all_similarities:
        max_sim = max(all_similarities)
        min_sim = min(all_similarities)
        avg_sim = sum(all_similarities) / len(all_similarities)
        print(f"\nSimilarity statistics:")
        print(f"  Highest similarity: {max_sim:.3f}")
        print(f"  Lowest similarity: {min_sim:.3f}")
        print(f"  Average similarity: {avg_sim:.3f}")
        print(f"  Current threshold: {similarity_threshold:.3f}")
    # Sort by color ratio (for color queries)
    if "yellow" in text_query.lower() or "red" in text_query.lower() or "blue" in text_query.lower() or "green" in text_query.lower() or "white" in text_query.lower() or "black" in text_query.lower():
        matched_boxes = sorted(matched_boxes, key=lambda x: x[2], reverse=True)  # Sort by color ratio
    else:
        matched_boxes = sorted(matched_boxes, key=lambda x: x[1], reverse=True)  # Sort by similarity
    if not matched_boxes:
        print(f"\nNo matching objects found")
        print("Suggestion: Try lowering the threshold or using more general descriptive terms")
        return None
    print(f"\nFound {len(matched_boxes)} matching objects")

    # Draw detection results
    for idx, (box, similarity, color_ratio) in enumerate(matched_boxes[:num]):
        x1, y1, x2, y2 = box
        # Draw detection box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Create label text
        label_text = f"No.{idx+1}: {color_ratio:.1%}"
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        # Draw label background
        cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
        # Draw label text
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image

if __name__ == '__main__':
    # Main program
    image_path = "pictures/cars.png"  
    text_query = "yellow car"
    print(f"Loading image: {image_path}")
    print(f"Search target: {text_query}")
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
    else:
        print(f"Image loaded successfully, size: {image.shape}")
        # Perform object detection - use more relaxed parameters
        result_image = detect_objects_with_text(
            image, 
            text_query, 
            similarity_threshold=0.2,   # Lower CLIP similarity threshold
            color_threshold=0.05,       # Lower color ratio threshold (5%+ yellow pixels)
            num=3                       # Only show the most blue car
        )
        if result_image is not None:
            print("Detection completed, displaying results...")
            # Display final results
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Result: {text_query}")
            plt.axis("off")
            plt.show()
        else:
            print("No matching objects detected")