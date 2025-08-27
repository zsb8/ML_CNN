import torch
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import cv2
import numpy as np

# Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load YOLO model
yolo_model = YOLO('yolov8n.pt').to('cpu')  # Use YOLOv8 model
# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Natural language object detection function
def detect_objects_with_text(image, text_query, confidence_threshold=0.5, similarity_threshold=0.28, num=1):
    # Use YOLO for object detection
    detections = yolo_model(image)
    boxes = detections[0].boxes.xyxy.cpu().numpy()  # Get bounding box coordinates
    scores = detections[0].boxes.conf.cpu().numpy()  # Get confidence scores
    categories = detections[0].boxes.cls.cpu().numpy().astype(int)  # Get category indices
    
    # Filter low confidence boxes
    valid_boxes = []
    for i, score in enumerate(scores):
        if score > confidence_threshold:
            x1, y1, x2, y2 = map(int, boxes[i])
            cropped_img = image[y1:y2, x1:x2]  # Crop the bounding box region
            valid_boxes.append((cropped_img, (x1, y1, x2, y2)))
    # Use CLIP to match text with candidate boxes
    matched_boxes = []
    for cropped_img, box_coords in valid_boxes:
        # Convert bounding box region to CLIP processable format
        clip_input = clip_processor(images=[cropped_img], return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(**clip_input)
            text_input = clip_processor(text=[text_query], return_tensors="pt").to(device)
            text_features = clip_model.get_text_features(**text_input)

        # Calculate similarity between image and text features
        similarity = torch.cosine_similarity(image_features, text_features).item()
        if similarity > similarity_threshold:  # Set matching threshold
            matched_boxes.append((box_coords, similarity))
    
    matched_boxes = sorted(matched_boxes, key=lambda x: x[1], reverse=True)

    if not matched_boxes:
        return None

    # Draw detection results
    for box, similarity in matched_boxes[:num]:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{text_query}, similarity = {similarity:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image


def process_video(input_video_path, output_video_path, text_query, confidence_threshold=0.5, similarity_threshold=0.25, num=1):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    out = None  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Call object detection function on each frame
        detected_frame = detect_objects_with_text(frame, text_query, confidence_threshold, similarity_threshold, num)
        # If no target is detected (return value is None), skip this frame
        if detected_frame is None:
            continue
        if out is None:
            height, width, _ = detected_frame.shape
            out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
        # Write to output video
        out.write(detected_frame)
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# Call function to process video
input_video_path = 'Video_data/Animal_600s_compressed.mp4'  
output_video_path = 'Video_data/Animal_filtered.mp4'  
text_query = 'dog'  
process_video(input_video_path, output_video_path, text_query)