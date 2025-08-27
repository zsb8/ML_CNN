import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import time

def load_train_data():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet requires 224x224 input
        transforms.Grayscale(3),  # Convert to 3 channels for AlexNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
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
        batch_size=32,  
        shuffle=True,
        num_workers=0
    )
    return trainloader

def create_model():
    # Load pre-trained AlexNet
    alexnet = models.alexnet(pretrained=True)
    
    # Modify the classifier for MNIST (10 classes)
    alexnet.classifier[6] = nn.Linear(4096, 10)
    
    return alexnet

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
        
        avg_loss = running_loss / len(trainloader)
        
        net.eval() 
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
