import os
from ultralytics import YOLO
from PIL import Image

def test_with_same_data():
    model_path = "./runs/quick_train/weights/best.pt"
    model = YOLO(model_path)
    images_dir = "./test_images"
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    image_files.sort()
    print(f"Found {len(image_files)} images")
    for i, img_file in enumerate(image_files[:len(image_files)]):
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        print(f"\n--- Test image {i+1}: {img_file} ---")
        results = model.predict(
            source=img,
            conf=0.1,      # Low confidence threshold
            imgsz=640,     # Image size
            device='cpu', 
            verbose=True  # Show detailed information
        )
        result = results[0]

        if result.boxes is not None:
            print(f"Found {len(result.boxes)} Objects:")
            for j, box in enumerate(result.boxes):
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = result.names[cls]
                print(f"  Object {j+1}: {class_name} (Confidence: {conf:.2f})")
        else:
            print("Can't find any objects")
        
        # Save result image
        save_path = f"./out_yolo/same_data_test_{i}.jpg"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(filename=save_path)
        print(f"Result saved: {save_path}")

if __name__ == "__main__":
    test_with_same_data()
