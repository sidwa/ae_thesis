"""
Performs reconstruction on given input on different AE types
in a grid for comparison
"""


import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision
from torchvision.utils import save_image, make_grid

from PIL import Image
import os

from modules import VectorQuantizedVAE, to_scalar, VAE, AAE
import vqvae
import aae
import vae

DEVICE="cuda:1"

#data_folder = "/shared/kgcoe-research/mil/ImageNet/"
data_folder = "./data/face/"
im_transform = transforms.Compose([ \
					 transforms.CenterCrop(128), \
					 transforms.ToTensor()])

test_dataset = datasets.ImageFolder(root=data_folder, transform=im_transform)
#test_dataset = datasets.ImageFolder(root=os.path.join(data_folder, "val"), transform=im_transform)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False)


# def get_recons(loader, hidden_size = 256):

# 		if model == "vqvae":
# 			model = VectorQuantizedVAE(3, hidden_size, 512).to(DEVICE)
# 			ckpt = torch.load("./models/imagenet/hs_{}/best.pt".format(hidden_size))
# 		elif model == "vae":
# 			model = VAE(3, hidden_size, hidden_size).to(DEVICE)
# 			ckpt = torch.load("./models/imagenet/hs_128_256_vae.pt")
# 		elif model == "aae":
# 			model = AAE(128, 3, hidden_size).to(DEVICE)
# 			ckpt = torch.load("./models/imagenet/aae/imagenet_hs_128_{}/best.pt".format(hidden_size))
		
# 		model.load_state_dict(ckpt)

# 		args = type('', (), {})()
# 		args.device=DEVICE

# 		gen_img, _ = next(iter(loader))

# 		# grid = make_grid(gen_img.cpu(), nrow=8)
# 		# torchvision.utils.save_image(grid, "hs_{}_recons.png".format(hidden_size))
# 		#exit()

# 		reconstruction = generate_samples(gen_img, model, args)
# 		grid = make_grid(reconstruction.cpu(), nrow=8)

# 		return grid

def get_vqvae_recons(loader, hidden_size = 256):

		model = VectorQuantizedVAE(3, hidden_size, 512).to(DEVICE)
		ckpt = torch.load("./models/imagenet/best.pt".format(hidden_size))
		model.load_state_dict(ckpt)

		args = type('', (), {})()
		args.device=DEVICE

		gen_img, _ = next(iter(loader))

	   	# grid = make_grid(gen_img.cpu(), nrow=8)
	   	# torchvision.utils.save_image(grid, "hs_{}_recons.png".format(hidden_size))
	   	#exit()

		reconstruction = vqvae.generate_samples(gen_img, model, args)
		grid = make_grid(reconstruction.cpu(), nrow=8)

		return grid

def get_vae_recons(loader, hidden_size = 256):

		model = VAE(3, hidden_size, hidden_size).to(DEVICE)

		ckpt = torch.load("./models/imagenet_hs_128_256_vae.pt")
		model.load_state_dict(ckpt)
		args = type('', (), {})()
		args.device=DEVICE
		gen_img, _ = next(iter(loader))
		# grid = make_grid(gen_img.cpu(), nrow=8)
		# torchvision.utils.save_image(grid, "hs_{}_recons.png".format(hidden_size))
		#exit()

		reconstruction = vae.generate_samples(gen_img, model, args)
		grid = make_grid(reconstruction.cpu(), nrow=8)

		return grid

def get_aae_recons(loader, hidden_size = 256):

		model = AAE(128, 3, hidden_size).to(DEVICE)

		ckpt = torch.load("./models/aae/imagenet_hs_128_{}/best.pt".format(hidden_size))

		model.load_state_dict(ckpt)

		args = type('', (), {})()
		args.device=DEVICE

		gen_img, _ = next(iter(loader))

	   # grid = make_grid(gen_img.cpu(), nrow=8)
	   # torchvision.utils.save_image(grid, "hs_{}_recons.png".format(hidden_size))
	   #exit()

		reconstruction = aae.generate_samples(gen_img, model, args)
		grid = make_grid(reconstruction.cpu(), nrow=8)

		return grid


models = [get_aae_recons, get_vae_recons, get_vqvae_recons] 
hidden_size = 256

orig_img, _ = next(iter(test_loader))
img_grid = make_grid(orig_img, nrow=8)
for model in models:
	print(img_grid.shape)
	img_grid = torch.cat([img_grid, model(test_loader, hidden_size)], axis=1)


print(img_grid.shape)
#img_grid = get_recons(hidden_size=32)
torchvision.utils.save_image(img_grid, "hs_recons_ae.png")