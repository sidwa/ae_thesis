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

from modules import VectorQuantizedVAE, to_scalar, VAE, ACAI, AutoEncoder
import train_util

from tensorboardX import SummaryWriter

def disp_img(img):
	# img[0] = img[0] * pixel_std[0] + pixel_mean[0]
	# img[1] = img[1] * pixel_std[1] + pixel_mean[1]
	# img[2] = img[2] * pixel_std[2] + pixel_mean[2]
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1,2,0)))
	plt.show()


class ImgClassifier(torch.nn.Module):

	def __init__(self, ae, hidden_size, resolution, num_classes):
		super().__init__()
		self.hidden_size = hidden_size
		self.net = ae
		self.resolution = resolution
		if self.net is not None:
			if "VectorQuantizedVAE" in str(type(self.net)):
				self.ae_type = "vqvae"
			elif "VAE" in str(type(self.net)):
				self.ae_type = "vae"
			elif "ACAI" in str(type(self.net)):
				self.ae_type = "acai"

			self.cl = torch.nn.Sequential(
				torch.nn.Linear( (self.resolution // 4) * (self.resolution // 4) * self.hidden_size, num_classes),
				torch.nn.Softmax())
		else:
			self.cl = torch.nn.Sequential(
				torch.nn.Linear(self.resolution*self.resolution*3,num_classes),
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
				elif self.ae_type == "acai":
					out = self.net.encoder(dat)
			#pdb.set_trace()
			out = out.view(-1, self.hidden_size* (self.resolution // 4) * (self.resolution // 4) )
		#print("2", out.shape)
		else:
			out = dat.view(-1, 3*self.resolution*self.resolution) # *********
		out = self.cl(out)

		return out

class SupervisedImgClassifier(torch.nn.Module):

	def __init__(self, hidden_size, enc_type, resolution, num_classes):
		super().__init__()
		self.hidden_size = hidden_size
		self.resolution = resolution
		
		self.net = AutoEncoder.get_encoder(3, hidden_size, args.enc_type)


		self.cl = torch.nn.Sequential(
			torch.nn.Linear( (self.resolution // 4) * (self.resolution // 4) * self.hidden_size, num_classes),
			torch.nn.Softmax())

	def forward(self, dat):

		out = self.net.encoder(dat)
		out = out.view(-1, self.hidden_size* (self.resolution // 4) * (self.resolution // 4) )
		out = self.cl(out)

		return out


def get_model(model, hidden_size, num_channels, resolution, enc_type, dec_type, num_classes, k=512):
	

	CKPT_DIR = f"models/{model}_{args.recons_loss}/{args.train_dataset}/depth_{enc_type}_{dec_type}_hs_{args.img_res}_{hidden_size}/best.pt"
	if model == "vqvae":
		#CKPT_DIR = "models/imagenet/hs_32_4/best.pt"#.format(hidden_size)
		model = VectorQuantizedVAE(num_channels, hidden_size, k, enc_type, dec_type)
	elif model == "vae":
		model = VAE(num_channels, hidden_size, enc_type, dec_type)
	elif model == "acai":
		model = ACAI(resolution, num_channels, hidden_size, enc_type, dec_type)
	else:
		model = None

	if model != "supervised":
		imgclassifier = ImgClassifier(model, hidden_size, resolution, num_classes)
	else:
		imgclassifier = SupervisedImgClassifier(hidden_size, enc_type, resolution, num_classes)

	if hidden_size > 0:
		ckpt = torch.load(CKPT_DIR)
		model.load_state_dict(ckpt["model"])

	return imgclassifier

	
def get_loaders(args):
	"""
		creates data loaders and returns the loaders as well as the resolution of image the loaders will return
	"""

	if args.dataset == "tiny-imagenet":
		resolution = args.img_res
		num_classes = 200
		im_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(resolution), 
												torchvision.transforms.ToTensor()])

		# train_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/train", transform=im_transform )
		# test_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/test", transform=im_transform )
		# val_dataset = torchvision.datasets.ImageFolder(root="./data/tiny-imagenet-200/val", transform=im_transform )
		if args.data_folder is not None:
			args.data_folder = args.data_folder
		else:
			args.data_folder = "../datasets/tiny-imagenet-200/"
		train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_folder, "train"), transform=im_transform )
		test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_folder, "test"), transform=im_transform )
		valid_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_folder, "val"), transform=im_transform )
		 

	elif args.dataset == "cifar10":
		resolution = args.img_res
		num_classes = 10
		im_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(resolution), 
												torchvision.transforms.ToTensor()])

		if args.data_folder is not None:
			args.data_folder = args.data_folder
		else:
			args.data_folder = "../datasets/cifar10/"
		train_dataset = torchvision.datasets.CIFAR10(root=args.data_folder, train=True, transform=im_transform, download=True)
		test_dataset = torchvision.datasets.CIFAR10(root=args.data_folder, train=False, transform=im_transform, download=True)
		valid_dataset = test_dataset

	elif args.dataset == "imagenet":
		resolution = args.img_res
		num_classes = 1000
		if args.data_folder is not None:
			args.data_folder = args.data_folder
		else:
			args.data_folder = "../datasets/ImageNet/"
		im_transform = torchvision.transforms.Compose([ \
						torchvision.transforms.CenterCrop(args.img_res), \
						torchvision.transforms.ToTensor()])
		train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_folder, "train"), transform=im_transform )
		test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.data_folder, "val"), transform=im_transform )
		valid_dataset = test_dataset
	else:
		print("invalid dataset")
		exit()

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

	return train_loader, test_loader, num_classes



def main(args, device):
	
	writer = SummaryWriter('./logs/imgclass/')

	train_loader, test_loader, num_classes = get_loaders(args)

	classifier = get_model(args.model, args.hidden_size, 3, args.img_res, args.enc_type, args.dec_type, num_classes)
	classifier = classifier.to(device)
	
	crit = torch.nn.CrossEntropyLoss()
	opt = torch.optim.Adam(classifier.parameters())

	max_test_acc = 0
	for epoch in range(args.num_epochs):

		# train an epoch
		run_loss = 0.0
		for (iter_num, (data, label)) in enumerate(train_loader):

			data = data.to(device)
			label = label.to(device)

			opt.zero_grad()

			logits = classifier(data)
			# pdb.set_trace()
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

	# General

	parser.add_argument('--model', default="acai", type=str, choices={"acai", "vae", "vqvae", "pcie"},
		help='name of the training dataset ')
	parser.add_argument('--train_dataset', default="tiny-imagenet", type=str, choices={"tiny-imagenet", "imagenet"},
		help='name of the training dataset ')
	parser.add_argument('--data_folder', type=str,
		help='name of the data folder')
	parser.add_argument('--dataset', default="imagenet", type=str, choices={"cifar10", "tiny-imagenet", "imagenet"},
		help='name of the dataset ')
	parser.add_argument('--img_res', default=128, type=int,
		help='image resolution to center crop')
	parser.add_argument("--input_type", type=str, default="image", choices={"image"},
		help='image resolution to random crop')

	# model
	parser.add_argument('weights', type=str, choices={"init", "load"}, default = "load",
		help='load pretrained weights of the network or start from scratch')
	parser.add_argument('--enc_type', type=str, choices={"shallow", "moderate_shallow", "moderate", "deep"},
		help='depth of encoder')
	parser.add_argument('--dec_type', type=str, choices={"shallow", "moderate_shallow", "moderate", "deep"},
		help='depth of decoder')
	parser.add_argument('--hidden_size', type=int, default=256,
		help='size of the latent vectors (default: 256)')
	parser.add_argument('--scale_factor', type=int, default=4,
		help='while computing latents  h,w are reduced by this factor (default: 4)')
	parser.add_argument("--recons_loss", type=str, 
		choices={"mse", "gan", "comp", "comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc"},
		help="type of reconstruction loss to use, discriminator n/w of GAN or mean square error")
	parser.add_argument("--prior_loss", type=str,
		choices={"kl_div", "gan"}, default="kl_div",
		help="force latents to fall in a prior distribution using this loss")

	
	# perturb factors

	# perturbation types  
	for perturb_type in train_util.perturb_dict:
		parser.add_argument(f"--{perturb_type}", action="store_true",
		help=train_util.perturb_dict[perturb_type].help)

	parser.add_argument("--recons_type", type=str, choices={"same", "reverse"}, default="same",
		help="same: reconstruction to match the noised/disturbed image\n \
			  reverse: reconstruction should match orignal image, reversing the perturbation")
	parser.add_argument("--perturb_feat_gan", action="store_true",
		help="choose if a discriminator should test if any info of perturbations can be extracted from latents")
	

	# Optimization
	parser.add_argument('--batch_size', type=int, default=60,
		help='batch size (default: 128)')
	parser.add_argument('--num_epochs', type=int, default=20,
		help='number of epochs (default: 100)')
	parser.add_argument('--lr', type=float, default=1e-5,
		help='learning rate for Adam optimizer (default: 1e-5)')
	parser.add_argument("--disc_lr", type=float, default=1e-5)
	parser.add_argument('--gamma', '-g', type=float, default=0.2,
		help='regularization for critic to perform well with bad reconstructions')
	parser.add_argument('--lam', '-l', type=float, default=0.5,
		help='weight for loss from critic')
	parser.add_argument("--threshold", type=float, default=0.1,
		help="val loss reduction threshold")
	parser.add_argument("--lr_patience", type=int, default=2,
		help="epochs of patience for val loss reduction under threshold before reducing learning rate")
	parser.add_argument("--stop_patience", type=int, default=5,
		help="epochs of patience for val loss reduction under threshold before early stop")

	# Miscellaneous
	parser.add_argument('--output_folder', type=str,
		help='name of the output folder')
	parser.add_argument('--num_workers', type=int, default=0,
		help='number of workers for trajectories sampling (default: 0)')
	parser.add_argument('--print_interval', type=int, default=1000,
		help='number of workers for trajectories sampling (default: 0)')
	parser.add_argument('--device', type=str, default='cuda', choices={"cpu", "cuda", "cuda:0", "cuda:1"},
		help='set the device (cpu or cuda, default: cpu)')

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	args = parser.parse_args()
	
	main(args, device)