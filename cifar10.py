
import numpy as np
import os
import sys
import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5, stride=1, padding=2)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5, stride=1, padding=2)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(16 * 32 * 32, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 1000)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 32 * 32)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def main():

	parser = argparse.ArgumentParser()

	parser.add_argument("--batch_size", type=int, default = 128)
	parser.add_argument("--num_epochs", type=int, default=10)

	args = parser.parse_args()
	print(f"batch:{args.batch_size}", file=sys.stderr)
	print(f"epochs:{args.num_epochs}", file=sys.stderr)
	sys.stdout.flush()
	batch_size = args.batch_size
	im_transform = transforms.Compose([ \
								transforms.CenterCrop(128), \
								transforms.ToTensor()])

	data_folder = "../datasets/ImageNet"
	train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "train"), transform=im_transform )
	test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "val"), transform=im_transform )
	valid_dataset = test_dataset


	# transform = transforms.Compose(
	#     [transforms.ToTensor(),
	#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	# train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True,
	#                                     download=True, transform=transform)

	# test_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=False,
	#                                    download=True, transform=transform)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
											shuffle=True, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
											shuffle=False, num_workers=0)

	device = torch.device("cuda")


	#net = Net()
	net = torchvision.models.resnet34()
	if torch.cuda.device_count() > 1:
		net = nn.DataParallel(net)
	net.to(device)
	
	train(net, train_loader, args.num_epochs, device)
	test(net, test_loader)

def train(net, trainloader, epochs, device):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	for epoch in range(epochs):  # loop over the dataset multiple times

		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			inputs = inputs.to(device)
			labels = labels.to(device)

			# forward + backward + optimize
			try:
				outputs = net(inputs)
			except RuntimeError as error:
				print(error, file=sys.stderr)
				exit()

			print(f"output:{outputs.shape}\t labels:{labels.shape}", file=sys.stderr)
			sys.stderr.flush()
			ce_loss = F.cross_entropy(outputs, labels)
			# mse_loss = F.mse_loss(outputs, labels)
			loss = ce_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			if True:#i % 10 == i:    # print every 100 mini-batches
				print('[%d, %5d] loss: %f' %
					(epoch + 1, i + 1, running_loss / 2000))
				sys.stdout.flush()
				sys.stderr.flush()
				running_loss = 0.0
				#break

			print(torch.cuda.max_memory_allocated(torch.device("cuda:0")), file=sys.stderr)
			sys.stderr.flush()

	print('Finished Training')

def test(net, testloader):
	correct = 0
	total = 0
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()

	print('Accuracy of the network on the 10000 test images: %d %%' % (
		100 * correct / total))


if __name__=="__main__":
	main()