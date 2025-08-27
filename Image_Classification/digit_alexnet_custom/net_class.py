import torch.nn as nn

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# AlexNet-style features for 32x32 input and 1 channel (MNIST)
		# Channels per conv layer: 96, 256, 384, 384, 256
		self.features = nn.Sequential(
			nn.Conv2d(1, 96, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),  # 32 -> 16
			nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),  # 16 -> 8
			nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)   # 8 -> 4
		)

		self.classifier = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(256 * 4 * 4, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(0.5),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 10)
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x