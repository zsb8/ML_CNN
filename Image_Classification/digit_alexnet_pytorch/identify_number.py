import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use device: {device}")

def create_model():
    # Load pre-trained AlexNet
    alexnet = models.alexnet(pretrained=True)
    
    # Modify the classifier for MNIST (10 classes)
    alexnet.classifier[6] = nn.Linear(4096, 10)
    
    return alexnet

net = create_model().to(device)
net.load_state_dict(torch.load('best_model.pth', map_location=device))
net.eval()  #

def load_picture(image_path='pictures_number/number_9_color.png'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet requires 224x224 input
        transforms.Grayscale(3),  # Convert to 3 channels for AlexNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    try:
        image = Image.open(image_path).convert('L')
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"error: {e}")
        return None

def identify_number(image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = net(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)
        return predicted.item(), probabilities[0]

if __name__ == '__main__':
    print("=" * 50)
    image_tensor = load_picture()
    if image_tensor is None:
        print("not inout picture")
        exit()

    predicted_number, probabilities = identify_number(image_tensor)
    
    print(f"\n Indetify result:")
    print(f"result is the number: {predicted_number}")
    print(f"Probability: {probabilities[predicted_number].item():.2%}")
    print(f"\n First 3 ones:")
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    for i in range(3):
        num = sorted_indices[i].item()
        prob = sorted_probs[i].item()
        print(f"Order {i+1}: Number {num} (Porbility: {prob:.2%})")
    print("\n" + "=" * 50)