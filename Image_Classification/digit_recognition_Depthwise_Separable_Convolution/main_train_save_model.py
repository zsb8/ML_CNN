import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time

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

def load_train_data():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Smaller input size for faster training
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Simple normalization
    ])
    print("Load dataset...")
    trainset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=False,
        transform=transform
    )
    total_train_size = len(trainset)
    print(f"Total training samples: {total_train_size}")
    print(f"Using all {total_train_size} samples (100%) for training")
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,  # Larger batch size for faster training
        shuffle=True,
        num_workers=0
    )
    return trainloader

def create_model():
    return LightweightNet(num_classes=10)

def train_model(trainloader):
    net = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    epochs = 2  # training times
    best_accuracy = 0
    best_epoch = 0
    print("Begin training model...")
    start_time = time.time()
    
    for epoch in range(epochs):
        net.train()  # Use dropout
        running_loss = 0.0
        epoch_start_time = time.time()
        
        for i, data in enumerate(trainloader):
            inputs, labels = (data[0], data[1])
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 100 == 99:
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
        
        # 计算整个epoch的平均损失
        avg_loss = running_loss / len(trainloader)
        
        # 在训练集上计算准确率
        net.eval()  # 切换到评估模式
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data[0], data[1]
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"Average training loss: {avg_loss:.3f}")
        print(f"Training accuracy: {accuracy:.2f}%")
        
        # 保存准确率最高的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch + 1
            torch.save(net.state_dict(), 'best_model.pth')
            print(f"New best model saved! Accuracy: {best_accuracy:.2f}%")
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch cost: {epoch_time/60:.2f} minutes")

    total_time = time.time() - start_time
    print(f"\nFinish training! Total cost: {total_time/60:.2f} minutes")
    print(f"Best accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
    print("Best model saved as 'best_model.pth'")

if __name__ == '__main__':
    trainloader = load_train_data()
    train_model(trainloader)
