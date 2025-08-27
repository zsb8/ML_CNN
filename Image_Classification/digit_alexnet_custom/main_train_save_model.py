import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
from net_class import Net

def load_train_data():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
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
        batch_size=512,
        shuffle=True,
        num_workers=0
    )
    return trainloader

def train_model(trainloader):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    epochs = 5  # training times
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
        
        # Calculate the average loss of the entire epoch
        avg_loss = running_loss / len(trainloader)
        
        # Calculate the accuracy on the training set
        net.eval()  # Switch to evaluation mode
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
        
        # Save the model with the highest accuracy
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
