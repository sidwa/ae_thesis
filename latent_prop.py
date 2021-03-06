"""
	Program to find the distribution of latents for images of different classes.
	In simple words find which latent vectors from the codebook tend to get selected
	for different classes of images in dataset.
"""


import os
import pdb
import shutil
import glob
import argparse
from tqdm import tqdm
import pdb

import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE, to_scalar, VAE
from tensorboardX import SummaryWriter

def disp_img(img):
	# img[0] = img[0] * pixel_std[0] + pixel_mean[0]
	# img[1] = img[1] * pixel_std[1] + pixel_mean[1]
	# img[2] = img[2] * pixel_std[2] + pixel_mean[2]
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()


class ImgnetClassifier(torch.nn.Module):

	def __init__(self, ae, hidden_size):
		super().__init__()
		self.hidden_size = hidden_size
		self.net = ae

		if "VectorQuantizedVAE" in str(type(self.net)):
			self.ae_type = "vqvae"
			self.cl = torch.nn.Sequential(
				# torch.nn.Linear(8*8*self.hidden_size,10),
				torch.nn.Linear(32*32*3,10),
			)
		elif "VAE" in str(type(self.net)):
			self.ae_type = "vae"
			self.cl = torch.nn.Sequential(
				# torch.nn.Linear(8*8*self.hidden_size,10),
				torch.nn.Linear(32*32*3,10),
			)

	def forward(self, dat):
		with torch.no_grad():
			_, out, _ = self.net(dat)
				#print(out.shape)
				#pdb.set_trace()
		#pdb.set_trace()
		out = out.view(-1, self.hidden_size*8*8)
		#print("2", out.shape)
		#out = dat.view(-1, 3*32*32) # *********
		out = self.cl(out)

		return out

def get_model(ae_type, hidden_size, k, num_channels):
	CKPT_DIR = "models/imagenet/hs_32_4/best.pt"#.format(hidden_size)
	model = VectorQuantizedVAE(num_channels, hidden_size, k)
	imgnetclassifier = ImgnetClassifier(model, hidden_size)

	ckpt = torch.load(CKPT_DIR)
	model.load_state_dict(ckpt)

	return imgnetclassifier

def get_loaders(batch_size, data_folder):

	im_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(32), 
												torchvision.transforms.ToTensor()])

	# train_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=im_transform )

	# test_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/test", transform=im_transform )

	# val_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=im_transform )

	# train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "train"), transform=im_transform )
	# test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "val"), transform=im_transform )
	# valid_dataset = test_dataset

	train_dataset = torchvision.datasets.CIFAR10(root=data_folder, train=True, transform=im_transform )
	test_dataset = torchvision.datasets.CIFAR10(root=data_folder, train=False, transform=im_transform )
	valid_dataset = test_dataset


	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	return train_loader, test_loader



def main(args, device):
	
	writer = SummaryWriter('./logs/imgnet_rep/')
	classifier = get_model("vqvae", args.hidden_size, args.k, args.num_channels)
	classifier = classifier.to(device)

	train_loader, test_loader = get_loaders(args.batch_size, args.data_folder)
	
	crit = torch.nn.CrossEntropyLoss()
	opt = torch.optim.Adam(classifier.parameters())

	
	for epoch in range(args.epoch):

		# train an epoch
		run_loss = 0.0
		for (iter_num, (data, label)) in enumerate(train_loader):

			data = data.to(device)
			label = label.to(device)

			opt.zero_grad()

			logits = classifier(data)

			loss = crit(logits, label)
			loss.backward()
			
			opt.step()

			run_loss += loss.item()

			writer.add_scalar("loss/train", loss.item(), epoch * len(train_loader) + iter_num)

			if iter_num % 100 == 0:
				_, pred = torch.max(logits.data, 1)
				correct = (pred == label).sum().item()
				print(f"iter: {iter_num} :: loss: {run_loss/1000}")
				print(f"100 mini-batch accuracy: {correct/label.size(0)}")
				run_loss = 0.0

		#test acc
		total = 0
		correct = 0
		for data, label in test_loader:
			
			data = data.to(device)
			label = label.to(device)

			with torch.no_grad():
				outs = classifier(data)
				test_loss = crit(outs, label)
				_, predicted = torch.max(outs.data, 1)
				total += label.size(0)
				correct += (predicted == label).sum().item()
				writer.add_scalar("loss/test", test_loss, epoch)
		print(f"test set accuracy: {correct/total}")
		



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='imagenet classifier test for imagenet trained vqvae encoder')

	parser.add_argument("--hidden_size", "-hs", default=4, type=int, help="dim of latent vectors in codebook")
	parser.add_argument("--epoch", "-e", default=20, type=int, help="num epochs to train")
	parser.add_argument("-k", default=512, type=int, help="number of latent vectos in codebook")
	parser.add_argument("--batch_size", "-b", default=64, type=int, help="batch size")
	parser.add_argument("--data_folder", "-d", default="data/cifar10/", type=str)
	parser.add_argument("--num_channels", "-c", default=3, type=int, 
		help="don't bother if using color image dataset, switch to 1 for mnist and the like")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	args = parser.parse_args()
	
	main(args, device)