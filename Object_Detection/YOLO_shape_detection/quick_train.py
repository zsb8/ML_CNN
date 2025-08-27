from ultralytics import YOLO


def train_yolo_model(yaml_path="./data/dataset.yaml"):
    # Initialize model
    model = YOLO("yolov8n.pt")
    # Training parameters
    train_args = {
        'data': yaml_path,         # Use prepared config file
        'epochs': 5,               # Number of training epochs
        'imgsz': 640,              # Image size
        'batch': 8,                # Batch size
        'device': 'cpu',           # Use CPU
        'lr0': 0.01,               # Learning rate
        'patience': 10,            # Early stopping patience
        'save': True,              # Save model
        'project': './runs',       # Project directory
        'name': 'quick_train',     # Experiment name
        'verbose': True            # Show detailed information
    }
    try:
        # Start training
        print("Starting YOLO model training...")
        results = model.train(**train_args)
        print(f"✅ Training completed! Model saved at: {results.save_dir}")
        # Show training results
        print(f"📊 Training results:")
        print(f"   - Best model: {results.save_dir}/weights/best.pt")
        print(f"   - Latest model: {results.save_dir}/weights/last.pt")
        print("\n=== Summary ===")
        print("✅ YOLO format dataset preparation completed")
        print("✅ Model training completed")
        print("📁 Training results: ./runs/quick_train/")
        print("📁 Dataset: ./data/images/ and ./data/labels/")
        print("💡 You can now use the trained model for inference")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("Please check dataset configuration and training parameters")

if __name__ == "__main__":
    train_yolo_model()

