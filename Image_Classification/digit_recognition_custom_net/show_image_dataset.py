import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    img = img / 2 + 0.5 
    npimg = img.numpy() 
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

data_dir = r"D:\test1\data"

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = torchvision.datasets.MNIST(
    root=data_dir, train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

dataiter = iter(testloader)
images, labels = next(dataiter)

print("Real picture: ", " ".join(f"{labels[j]}" for j in range(4)))
imshow(torchvision.utils.make_grid(images))
# for i in range(4):
#     imshow(images[i])
