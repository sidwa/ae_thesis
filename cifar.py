"""
	Imagenet classifier network to check classification performance of encoder pre-trained 
	using vq-vae reconstruction on imagenet data itself.
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

from modules import VectorQuantizedVAE, to_scalar, VAE, AAE
from tensorboardX import SummaryWriter

def disp_img(img):
	# img[0] = img[0] * pixel_std[0] + pixel_mean[0]
	# img[1] = img[1] * pixel_std[1] + pixel_mean[1]
	# img[2] = img[2] * pixel_std[2] + pixel_mean[2]
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()


class ImgnetClassifier(torch.nn.Module):

	def __init__(self, ae, hidden_size, resolution, num_classes):
		super().__init__()
		self.hidden_size = hidden_size
		self.net = ae

		if self.net is not None:
			if "VectorQuantizedVAE" in str(type(self.net)):
				self.ae_type = "vqvae"
			elif "VAE" in str(type(self.net)):
				self.ae_type = "vae"
			elif "AAE" in str(type(self.net)):
				self.ae_type = "aae"

			self.cl = torch.nn.Sequential(
				torch.nn.Linear( (resolution / 4) * (resolution / 4) *self.hidden_size, num_classes),
				torch.nn.Softmax())
		else:
			self.cl = torch.nn.Sequential(
				torch.nn.Linear(32*32*3,num_classes),
				torch.nn.Softmax())
		

	def forward(self, dat):
		if self.hidden_size > 0:
			with torch.no_grad():
				if self.ae_type == "vqvae":
					_, _, out = self.net(dat)
				elif self.ae_type == "vae":
					mu, logvar = self.net.encoder(dat).chunk(2, dim=1)
					# define normal dist
					gauss = torch.distributions.Normal(mu, logvar.mul(.5).exp())
					out = gauss.sample()
				elif self.ae_type == "aae":
					out = self.net.encoder(dat)
			#pdb.set_trace()
			out = out.view(-1, self.hidden_size* (resolution / 4) * (resolution / 4) )
		#print("2", out.shape)
		else:
			out = dat.view(-1, 3*resolution*resolution) # *********
		out = self.cl(out)

		return out

def get_model(model, hidden_size, k, num_channels, resolution, num_classes):
	
	if model == "vqvae":
		if hidden_size==256:
			CKPT_DIR = "models/imagenet/best.pt" 
		elif hidden_size==128:
			CKPT_DIR = "models/imagenet/hs_{}/best.pt".format(hidden_size)
		else:
			CKPT_DIR = "models/imagenet/hs_{}/best.pt".format(hidden_size)
			#CKPT_DIR = "models/imagenet/hs_32_4/best.pt"#.format(hidden_size)
		model = VectorQuantizedVAE(num_channels, hidden_size, k)
	elif model == "vae":
		CKPT_DIR = f"models/imagenet_hs_128_{hidden_size}_vae.pt"
		model = VAE(num_channels, hidden_size, hidden_size)
		
	elif model == "aae":
		CKPT_DIR = f"models/aae/imagenet_hs_32_{hidden_size}/best.pt"
		model = AAE(32, num_channels, hidden_size)
	else:
		model = None

	imgnetclassifier = ImgnetClassifier(model, hidden_size, resolution)

	if hidden_size > 0:
		ckpt = torch.load(CKPT_DIR)
		model.load_state_dict(ckpt)

	return imgnetclassifier

	
def get_loaders(batch_size, dataset, data_folder):
	"""
		creates data loaders and returns the loaders as well as the resolution of image the loaders will return
	"""

	if dataset == "imagenet":
		resolution = 128
		num_classes = 100
		im_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(resolution), 
												torchvision.transforms.ToTensor()])

		train_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=im_transform )

		test_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/test", transform=im_transform )

		val_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=im_transform )

		train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "train"), transform=im_transform )
		test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "val"), transform=im_transform )
		valid_dataset = test_dataset
		 

	elif dataset == "cifar":
		resolution = 32
		num_classes = 10
		im_transform = torchvision.transforms.Compose([#torchvision.transforms.CenterCrop(32), 
												torchvision.transforms.ToTensor()])

		train_dataset = torchvision.datasets.CIFAR10(root=data_folder, train=True, transform=im_transform )
		test_dataset = torchvision.datasets.CIFAR10(root=data_folder, train=False, transform=im_transform )
		valid_dataset = test_dataset


		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
	else:
		print("invalid dataset")
		exit()

	return train_loader, test_loader, resolution, num_classes



def main(args, device):
	
	writer = SummaryWriter('./logs/imgnet_rep/')

	train_loader, test_loader, resolution, num_classes = get_loaders(args.batch_size, args.dataset,args.data_folder)

	classifier = get_model(args.model, args.hidden_size, args.k, args.num_channels, resolution, num_classes)
	classifier = classifier.to(device)
	
	crit = torch.nn.CrossEntropyLoss()
	opt = torch.optim.Adam(classifier.parameters())

	max_test_acc = 0
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
		max_test_acc = correct/total if correct/total > max_test_acc else max_test_acc
		print(f"test set accuracy: {correct/total}")
		
	print(f"Max test set accuracy: {max_test_acc}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='cifar10 classifier test for imagenet trained vqvae encoder')

	parser.add_argument("dataset", type=str, help="Dataset to run classification test on cifar|imagenet")
	parser.add_argument("--data_folder", "-d", default="data/cifar10/", type=str)
	parser.add_argument("--model", "-m", default="vqvae", type=str, help="type of autoencoder (vqvae|vae|aae)")
	parser.add_argument("--hidden_size", "-hs", default=256, type=int, help="dim of latent vectors in codebook, set to 0 to evaluate baseline(one layer MLP) scores")
	parser.add_argument("--epoch", "-e", default=20, type=int, help="num epochs to train")
	parser.add_argument("-k", default=512, type=int, help="number of latent vectos in codebook")
	parser.add_argument("--batch_size", "-b", default=64, type=int, help="batch size")
	parser.add_argument("--num_channels", "-c", default=3, type=int, 
		help="don't bother if using color image dataset, switch to 1 for mnist and the like")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	args = parser.parse_args()
	
	main(args, device)