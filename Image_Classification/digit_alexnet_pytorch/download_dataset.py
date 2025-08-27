import os
import torchvision
import torchvision.transforms as transforms

def download_mnist_dataset(save_path='./data'):
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download training dataset， train-images-idx3-ubyte：包含所有60,000个训练图像
    train_dataset = torchvision.datasets.MNIST(
        root=save_path,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test dataset， t10k-images-idx3-ubyte：包含所有10,000个测试图像
    test_dataset = torchvision.datasets.MNIST(
        root=save_path,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"Dataset downloaded successfully to {save_path}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_dataset, test_dataset

# Usage example
if __name__ == "__main__":
    save_path='D:/test1/data'
    train_dataset, test_dataset = download_mnist_dataset(save_path)