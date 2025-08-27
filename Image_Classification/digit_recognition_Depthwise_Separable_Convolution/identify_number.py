import torch
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Use device: {device}")

class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, 
                                  padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LightweightNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LightweightNet, self).__init__()
        
        # Feature extraction layers with depthwise separable convolutions
        self.features = nn.Sequential(
            # First block: 1 -> 32 channels
            DepthwiseSeparableConv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block: 32 -> 64 channels
            DepthwiseSeparableConv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block: 64 -> 128 channels
            DepthwiseSeparableConv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block: 128 -> 256 channels
            DepthwiseSeparableConv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def create_model():
    return LightweightNet(num_classes=10)

net = create_model().to(device)
net.load_state_dict(torch.load('best_model.pth', map_location=device))
net.eval()  #

def load_picture(image_path='pictures_number/number_2.png'):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Smaller input size for faster training
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Simple normalizatio
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