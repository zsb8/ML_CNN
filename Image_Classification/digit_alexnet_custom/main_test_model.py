import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from net_class import Net
device = torch.device("cpu")

def load_test_data():
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    print("Load dataset...")

    testset = torchvision.datasets.MNIST(
        root='D:/test1/data',
        train=False,
        download=False,
        transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )    
    return testloader

def test_model(testloader):
    net = Net().to(device)
    net.load_state_dict(torch.load('best_model.pth'))
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = (data[0], data[1])
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    final_accuracy = 100 * correct / total
    print(f"\n Accuracy of test is : {final_accuracy:.2f}%")

if __name__ == '__main__':
    testloader = load_test_data()
    test_model(testloader)
