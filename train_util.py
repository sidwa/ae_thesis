import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import sys
import argparse
import pdb
from PIL import ImageFilter
import random
from collections import namedtuple

import scipy

import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F

"""
perturb_function named tuple to encapsulate 
	-- func: perturb function, 

	-- conj: if perturbation function can be used in conjunction with other perturbation 
		two or more perturb functions with this value set as False cannot be used
		in conjunection for perturbation invariant latent training.
		But any number of functions with value set as True can be used in conjunction

	-- help: string that describe what it does, used in argparse
"""
perturb_nt = namedtuple("perturb_function", ["func", "num_class", "conj", "help"])

"""
	perturb_dict:
	key: perturbation feature
	value: namedtuple instance of perturb_nt
"""
perturb_dict = dict()

def rotate_perturb(image):
	angles = [0, 90, 180, 270]
	angle = random.choice(range(4))
	rot_img = transforms.functional.rotate(image, angles[angle])

	return rot_img, angle
perturb_dict["rotate_perturb"] = \
	perturb_nt(rotate_perturb, 4, False, "rotation: randomly rotates an image by (0, 90, 180, 270), non-conjugable")

def blur_perturb(image):

	if random.random() >= 0.5:
		blur_img = image.filter(ImageFilter.GaussianBlur(0.5))
		blur = 1.
	else:
		blur_img = image
		blur = 0.

	return blur_img, blur

perturb_dict["blur_perturb"] = \
	perturb_nt(blur_perturb, 2, True, "randomly applies gaussian blur on image (50% chance)")

def perturb_compatible(perturb_type):
	"""
		given a an iterable of types of perturbations
		to be performed, checks if all perturbations
		are compatible. In other words checks if there
		is only one perturb type with conj=False
	"""
	non_conjug = False
	for perturb in perturb_type:
		if not non_conjug and not perturb_dict[perturb].conj:
			non_conjug = True
		elif non_conjug and not perturb_dict[perturb].conj:
			return False
	else:
		return True

class custom_collate:
	"""
		Custom collate class which can be customized based on runtime arguments
	"""
	def __init__(self, perturb_type, img_res):
		self.perturbs = dict()
		
		self.img_res = img_res
		if not perturb_compatible(perturb_type):
			raise ValueError("Only one non-conjugable perturbation allowed")

		for perturb in perturb_type:
			try:
				self.perturbs[perturb] = perturb_dict[perturb].func
			except KeyError:
				raise ValueError(f"Perturbation operation:{perturb} not supported")
	
	def __call__(self, batch):
		"""
			works like a normal collate func; takes batch data as a list

			returns 
				torch.tensor batch data that maybe perturbed 
				torch.tensor original image
				dict(torch.tensor) labels where key corresponds to name 
					of perturbation type and the label corresponds to what amount
				dict(torch.tensor) labels where key corresponds to name 
					of perturbation type and the label corresponds to what amount(normalized)
		"""
		batch_img = []
		orig_batch_img = []
		labels = {}
		normalized_labels = {}
		for sample in batch:
			# sample_img = sample[0]
			# angle = 0
			sample_img = transforms.functional.center_crop(sample[0], self.img_res)
			for factor in self.perturbs:
				pert_sample_img, label = self.perturbs[factor](sample_img)
				if factor not in labels:
					labels[factor] = []
				labels[factor].append(label)
			
			orig_sample_img = transforms.functional.to_tensor(sample_img)
			orig_sample_img = torch.unsqueeze(orig_sample_img, dim=0)

			if len(self.perturbs) > 0:
				pert_sample_img = transforms.functional.to_tensor(pert_sample_img)
				pert_sample_img = torch.unsqueeze(pert_sample_img, dim=0)
				batch_img.append(pert_sample_img)
			else:
				batch_img.append(orig_sample_img)

			orig_batch_img.append(orig_sample_img)
		
		batch_img = torch.cat(batch_img)
		orig_batch_img = torch.cat(orig_batch_img)

		for perturb_type in labels:
			if perturb_dict[perturb_type].num_class > 2:
				dtype = torch.float
				norm_factor = perturb_dict[perturb_type].num_class
			elif perturb_dict[perturb_type].num_class == 2:
				dtype = torch.float
				norm_factor = 2

			labels[perturb_type] = torch.tensor(labels[perturb_type], dtype=dtype)
			normalized_labels[perturb_type] = labels[perturb_type] / (norm_factor - 1)

		return orig_batch_img, batch_img, labels, normalized_labels

def get_perturb_types(args):

	d_args = vars(args)
	num_perturb_types = 0
	perturb_types = []
	for perturb_type in perturb_dict:
		if d_args[perturb_type]:
			num_perturb_types += 1
			perturb_types.append(perturb_type)

	# print("pert!!:", perturb_types)
	# print("num:", len(perturb_types))
	return perturb_types

def get_dataloaders(args, perturb=[]):

	if args.dataset == "tiny-imagenet":
		# im_transform = torchvision.transforms.Compose([
		# 	torchvision.transforms.CenterCrop(args.img_res),
		# 	torchvision.transforms.ToTensor(),
		# 	#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		# ])
		if args.data_folder is not None:
			data_folder = args.data_folder
		else:
			data_folder = "../datasets/tiny-imagenet-200/"
		train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder,"train"))
		test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "test"))
		val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder ,"val"))
	elif args.dataset == "imagenet":
		# im_transform = torchvision.transforms.Compose([ \
		# 					torchvision.transforms.CenterCrop(args.img_res), \
		# 					torchvision.transforms.ToTensor()])
		if args.data_folder is not None:
			data_folder = args.data_folder
		else:
			data_folder = "../datasets/ImageNet/"
		train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "train"))
		test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "val"))
		val_dataset = test_dataset
	elif dataset == "cifar":
		resolution = args.img_res
		num_classes = 10
		im_transform = torchvision.transforms.Compose([torchvision.transforms.CenterCrop(resolution), 
												torchvision.transforms.ToTensor()])

		train_dataset = torchvision.datasets.CIFAR10(root=data_folder, train=True, transform=im_transform )
		test_dataset = torchvision.datasets.CIFAR10(root=data_folder, train=False, transform=im_transform )
		valid_dataset = test_dataset


	coll_func = custom_collate(perturb, args.img_res)

	# Define the data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, 
		pin_memory=True, collate_fn=coll_func)
	val_loader = torch.utils.data.DataLoader(val_dataset,
		batch_size=args.batch_size, shuffle=False, drop_last=True,
		pin_memory=True, collate_fn=coll_func)
	test_loader = torch.utils.data.DataLoader(test_dataset,
		batch_size=16, shuffle=False, collate_fn=coll_func)

	# train_loader = torch.utils.data.DataLoader(train_dataset,
	# 	batch_size=args.batch_size, shuffle=True,
	# 	num_workers=0, pin_memory=True)
	# val_loader = torch.utils.data.DataLoader(val_dataset,
	# 	batch_size=args.batch_size, shuffle=False, drop_last=True,
	# 	num_workers=0, pin_memory=True)
	# test_loader = torch.utils.data.DataLoader(test_dataset,
	# 	batch_size=16, shuffle=False)

	return train_loader, val_loader, test_loader

def get_legacy_dataloaders(args):

	if args.dataset == "tiny-imagenet":
		im_transform = torchvision.transforms.Compose([
			torchvision.transforms.CenterCrop(args.img_res),
			torchvision.transforms.ToTensor(),
			#torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		if args.data_folder is not None:
			data_folder = args.data_folder
		else:
			data_folder = "../datasets/tiny-imagenet-200/"
			train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder,"train"), )
			test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "test"))
			val_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder ,"val"))
	elif args.dataset == "imagenet":
		im_transform = torchvision.transforms.Compose([ \
							torchvision.transforms.CenterCrop(args.img_res), \
							torchvision.transforms.ToTensor()])
		if args.data_folder is not None:
			data_folder = args.data_folder
		else:
			data_folder = "../datasets/ImageNet/"
			train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "train"))
			test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_folder, "val"))
			val_dataset = test_dataset

	# Define the data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True,
		num_workers=0, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset,
		batch_size=args.batch_size, shuffle=False, drop_last=True,
		num_workers=0, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_dataset,
		batch_size=16, shuffle=False)

	return train_loader, val_loader, test_loader


def log_input_img_grid(test_loader, writer, nrow=8):
	"""
		get input images from test_loader to be reconstructed to test autoencoder training
	"""
	fixed_images = next(iter(test_loader))[0]
	fixed_grid = torchvision.utils.make_grid(fixed_images, nrow=nrow)
	writer.add_image('original', fixed_grid, 0)

	return fixed_images

def get_reconstruction(images, model, device):
	with torch.no_grad():
		images = images.to(device)
		output = model(images)
	return output[0]

def log_recons_img_grid(fixed_images, model, step, device, writer):
	nrow = fixed_images.shape[0]//2
	reconstruction = get_reconstruction(fixed_images, model, device)
	grid = torchvision.utils.make_grid(reconstruction.cpu(), nrow=nrow)
	writer.add_image('reconstruction', grid, step)

def save_recons_img_grid(split, fixed_images, model, step, args):
	model.eval()

	nrow = fixed_images.shape[0]//2
	x = fixed_images.to(args.device)
	out = model(x)
	x_tilde = out[0]
	x_cat = torch.cat([x, x_tilde], 0)
	images = x_cat.cpu().data

	save_dir = os.path.join("./logs", args.output_folder, f"{split}_recons")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	print(f"save_dir:{save_dir}", file=sys.stderr)
	torchvision.utils.save_image(
		images,
		os.path.join(save_dir, f"epoch_{step}.png"),
		nrow=nrow
	)

def log_interp_img_grid(fixed_images, model, step, device, writer):
	nrow = fixed_images.shape[0]//2

	out = model(fixed_images.to(device))
	x_tilde = out[0]
	z = out[1]
	x_tilde_interp = x_tilde[:nrow]
	x_tilde_1 = x_tilde[nrow:]
	interp_coeff = 0
	while interp_coeff < 1.1:
		alpha = torch.ones(nrow, 1, 1, 1).to(device) * interp_coeff
		z_interp = torch.lerp(z[:nrow], z[nrow:], alpha)
		x_tilde_interp = torch.cat([x_tilde_interp, model.decoder_forward(z_interp)])

		interp_coeff += 0.1
	
	x_tilde_interp = torch.cat([x_tilde_interp, x_tilde_1])

	grid = torchvision.utils.make_grid(x_tilde_interp.cpu(), nrow=nrow)
	writer.add_image(f"interpolated reconstruction{interp_coeff}", grid, step)

def save_interp_img_grid(split, fixed_images, model, step, args):
	
	nrow = fixed_images.shape[0]//2

	out = model(fixed_images.to(args.device))
	x_tilde = out[0]
	z = out[1]
	x_tilde_interp = x_tilde[:nrow]
	x_tilde_1 = x_tilde[nrow:]
	interp_coeff = 0
	while interp_coeff < 1.1:
		alpha = torch.ones(nrow, 1, 1, 1).to(args.device) * interp_coeff
		z_interp = torch.lerp(z[:nrow], z[nrow:], alpha)
		x_tilde_interp = torch.cat([x_tilde_interp, model.decoder_forward(z_interp)])

		interp_coeff += 0.1
	
	x_tilde_interp = torch.cat([x_tilde_interp, x_tilde_1])
	
	save_dir = os.path.join("./logs", args.output_folder, f"{split}_interp")
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	torchvision.utils.save_image(
		x_tilde_interp.cpu(),
		os.path.join(save_dir, f"epoch_{step}.png"),
		nrow=8
	)


def log_interpolation(model, images, device, writer):

	out, z = model(images)

	alphas = [0, 0.25, 0.5, 0.75, 1.0]
	batch_size = out[0].shape[0]
	alpha = torch.tensor(alphas * batch_size).reshape([-1, len(alphas)])

	alpha = torch.rand(out[1].shape[0], 1, 1, 1).to(device)
	z_interp = torch.lerp(z, swap_halves(z), alpha)
	x_tilde_interp = model.decoder_forward(z_interp)
	
	return x_tilde_interp, alpha

def log_losses(data_split, losses, step, writer):
	
	for loss in losses:
		writer.add_scalar(f"{data_split}/{loss}", losses[loss].item(), step)

def log_latent_metrics(data_split, latents, step, writer):

	writer.add_histogram(f"{data_split}/latent", latents, step)

def ae_data_parallel(model):
	"""
		dataparallel on composing networks. allowing AE to have
		extra functions while supporting data parallel.
	"""
	model.encoder = torch.nn.DataParallel(model.encoder)
	model.decoder = torch.nn.DataParallel(model.decoder)

	return model

REAL_LABEL = 0.
FAKE_LABEL = 1.

def get_recons_loss(images, x_tilde, device, loss_type, discriminators):

	
	if loss_type == "mse":
		recons_loss = F.mse_loss(x_tilde, images, reduction="sum") / images.shape[0]
	else:
		recons_disc = discriminators["recons_disc"][0]
		if loss_type == "gan":
			disc_labels = torch.ones(x_tilde.shape[0]) * REAL_LABEL
			disc_labels = disc_labels.to(device)

			disc_pred = torch.squeeze(recons_disc(x_tilde))
			recons_loss = F.binary_cross_entropy_with_logits(disc_pred, disc_labels)
		elif loss_type == "comp":
			disc_labels = torch.ones(x_tilde.shape[0]) * REAL_LABEL
			disc_labels = disc_labels.to(device)
			disc_input = torch.cat([images, x_tilde], dim=1)
			disc_pred = torch.squeeze(recons_disc(disc_input))
			recons_loss = F.binary_cross_entropy_with_logits(disc_pred, disc_labels)
		elif "comp_2" in loss_type:

			disc_input = torch.cat([images, x_tilde], dim=1)
			batch = images.shape[0]
			batch_idx = torch.arange(batch).reshape(-1,1)
			# randomly permute channels corresponding to real and reconstruction images
			channel_idx = torch.rand(batch, 1)
			channel_config = torch.where(channel_idx>=0.5, torch.tensor([1]), torch.tensor([0]))
			channel_idx = torch.where(channel_config==0, torch.tensor([[0,1,2,3,4,5]]), torch.tensor([3,4,5,0,1,2]))
			disc_input = disc_input[batch_idx, channel_idx, :, :].to(device)

			disc_pred = torch.squeeze(recons_disc(disc_input))

			# don't care
			if loss_type == "comp_2_dc":
				
				disc_labels = torch.ones(batch) * REAL_LABEL
				disc_labels = disc_labels.to(device)
				disc_pred = torch.squeeze(disc_pred[batch_idx, channel_config])
			# fully adversarial
			elif loss_type == "comp_2_adv":
				disc_labels = torch.where(channel_config==0, torch.tensor([FAKE_LABEL, REAL_LABEL]), torch.tensor([REAL_LABEL, FAKE_LABEL]))
				disc_labels = disc_labels.to(device)
			recons_loss = F.binary_cross_entropy_with_logits(disc_pred, disc_labels)
			
			
		elif "comp_6" in loss_type:
			
				batch = images.shape[0]
				batch_idx = torch.arange(batch).reshape(-1,1)
				disc_input = torch.cat([images, x_tilde], dim=1)
				channel_idx = torch.tensor([torch.randperm(6).tolist() for _ in range(batch)])
				disc_input = disc_input[batch_idx, channel_idx, :, :].to(device)

				if "color" in loss_type:
					color_pred, disc_pred = recons_disc(disc_input)
					color_labels = torch.where(channel_idx>=3, channel_idx-3, channel_idx).to(device)
					color_loss = F.cross_entropy(color_pred, color_labels)
				else:
					disc_pred = torch.squeeze(recons_disc(disc_input))			
				
				if "comp_6_dc" in loss_type:
					disc_labels = torch.ones([batch, 3], dtype=torch.long) * REAL_LABEL
					disc_labels = disc_labels.to(device)
					disc_pred = disc_pred[channel_idx>=3].reshape(batch,-1)
				elif "comp_6_adv" in loss_type:
					disc_labels = torch.where(channel_idx>=3, torch.tensor([REAL_LABEL]), torch.tensor([FAKE_LABEL]))
					disc_labels = disc_labels.to(device)
				recons_loss = F.binary_cross_entropy_with_logits(disc_pred, disc_labels)

				if "color" in loss_type:
					recons_loss += color_loss
		else:
			print(f"invalid recons loss type:{loss_type}")

	return recons_loss
	

def get_recons_disc_loss(loss_type, recons_disc, images, x_tilde, device):
	
	if loss_type == "gan":
		disc_input = torch.cat([images, x_tilde], dim=0)
		disc_labels = torch.cat([torch.ones(images.shape[0])*REAL_LABEL, torch.ones(x_tilde.shape[0])*FAKE_LABEL], dim=0)
		rand_idx = torch.randperm(disc_input.shape[0])
		disc_input = disc_input[rand_idx]
		disc_labels = disc_labels[rand_idx].to(device)

		disc_pred = torch.squeeze(recons_disc(disc_input))
		disc_loss = F.binary_cross_entropy_with_logits(disc_pred, disc_labels)
	elif loss_type=="comp":
		disc_anchor_input = torch.cat([images, images], dim=0)
		disc_recons_input = torch.cat([images, x_tilde], dim=0)
		disc_input = torch.cat([disc_anchor_input, disc_recons_input], dim=1)
		disc_labels = torch.cat([torch.ones(images.shape[0])*REAL_LABEL, torch.ones(x_tilde.shape[0])*FAKE_LABEL], dim=0)
		rand_idx = torch.randperm(disc_input.shape[0])
		disc_input = disc_input[rand_idx]
		disc_labels = disc_labels[rand_idx].to(device)

		disc_pred = torch.squeeze(recons_disc(disc_input))
		disc_loss = F.binary_cross_entropy_with_logits(disc_pred, disc_labels)
	elif "comp_2" in loss_type:

		disc_input = torch.cat([images, x_tilde], dim=1)
		real_disc_input = torch.cat([images, images], dim=1)
		recons_disc_input = torch.cat([x_tilde, x_tilde], dim=1)
		
		batch_size = images.shape[0]
		batch_idx = torch.arange(batch_size).reshape(-1,1)
		
		# randomly permute channels corresponding to real and reconstruction images
		channel_idx = torch.rand(batch_size, 1)
		channel_config = torch.where(channel_idx>=0.5, torch.tensor([1]), torch.tensor([0]))
		channel_idx = torch.where(channel_config==0, torch.tensor([3,4,5,0,1,2]), torch.tensor([[0,1,2,3,4,5]]))
		
		#add samples where both inputs real img or reconstructions
		disc_input = disc_input[batch_idx, channel_idx, :, :].to(device)
		disc_input = torch.cat([disc_input, real_disc_input, recons_disc_input], dim=0)

		disc_labels = torch.where(channel_config==0, torch.tensor([FAKE_LABEL, REAL_LABEL]), torch.tensor([REAL_LABEL, FAKE_LABEL]))
		real_disc_labels = torch.full((images.shape[0],2), REAL_LABEL)
		recons_disc_labels = torch.full((images.shape[0],2), FAKE_LABEL)

		disc_labels = torch.cat([disc_labels, real_disc_labels, recons_disc_labels], dim=0)
		disc_labels = disc_labels.to(device)

		disc_pred = torch.squeeze(recons_disc(disc_input))
		rand_idx = torch.randperm(disc_input.shape[0])	
		disc_pred = disc_pred[rand_idx]
		disc_labels = disc_labels[rand_idx]
		disc_loss = F.binary_cross_entropy_with_logits(disc_pred, disc_labels)
		
		
	elif "comp_6" in  loss_type:

		if "color" in loss_type:
			batch_size = images.shape[0]
			batch_idx = torch.arange(batch_size*3).reshape(-1,1)
			disc_input = torch.cat([images, x_tilde], dim=1)
			real_disc_input = torch.cat([images, images], dim=1)
			recons_disc_input = torch.cat([x_tilde, x_tilde], dim=1)

			#add samples where both inputs real img or reconstructions
			disc_input = torch.cat([disc_input, real_disc_input, recons_disc_input], dim=0)

			# randomly permute channels corresponding to real and reconstruction images
			channel_idx = torch.tensor([torch.randperm(6).tolist() for _ in range(batch_size*3)])
			disc_input = disc_input[batch_idx, channel_idx, :, :].to(device)

			# labels for real/recons channel predictions
			disc_chtype_labels = torch.where(channel_idx[:batch_size]>=3, torch.tensor([FAKE_LABEL]), torch.tensor([REAL_LABEL]))
			real_disc_labels = torch.full((images.shape[0],6), REAL_LABEL)
			recons_disc_labels = torch.full((images.shape[0],6), FAKE_LABEL)
			disc_chtype_labels = torch.cat([disc_chtype_labels, real_disc_labels, recons_disc_labels], dim=0)
			disc_chtype_labels = disc_chtype_labels.to(device)

			# labels for identifying color channel (R/G/B) or (0/1/2)
			disc_color_label = torch.where(channel_idx>=3, channel_idx-3, channel_idx).to(device)

			# random shuffle discriminator input in batch dimension
			rand_idx = torch.randperm(disc_input.shape[0])
			disc_chtype_labels = disc_chtype_labels[rand_idx]
			disc_input = disc_input[rand_idx]

			disc_color_pred, disc_chtype_pred = recons_disc(disc_input)
			
			disc_chtype_loss = F.binary_cross_entropy_with_logits(disc_chtype_pred, disc_chtype_labels)
			disc_color_loss = F.cross_entropy(disc_color_pred, disc_color_label)

			disc_loss = disc_chtype_loss + disc_color_loss
		else:
			batch_size = images.shape[0]
			batch_idx = torch.arange(batch_size).reshape(-1,1)
			disc_input = torch.cat([images, x_tilde], dim=1)
			real_disc_input = torch.cat([images, images], dim=1)
			recons_disc_input = torch.cat([x_tilde, x_tilde], dim=1)

			# randomly permute channels corresponding to real and reconstruction images
			channel_idx = torch.tensor([torch.randperm(6).tolist() for _ in range(batch_size)])
			disc_input = disc_input[batch_idx, channel_idx, :, :].to(device)

			#add samples where both inputs real img or reconstructions
			disc_input = torch.cat([disc_input, real_disc_input, recons_disc_input], dim=0)

			disc_labels = torch.where(channel_idx>=3, torch.tensor([FAKE_LABEL]), torch.tensor([REAL_LABEL]))
			real_disc_labels = torch.full((images.shape[0],6), REAL_LABEL)
			recons_disc_labels = torch.full((images.shape[0],6), FAKE_LABEL)
			disc_labels = torch.cat([disc_labels, real_disc_labels, recons_disc_labels], dim=0)

			disc_labels = disc_labels.to(device)

			disc_pred = torch.squeeze(recons_disc(disc_input))
			rand_idx = torch.randperm(disc_input.shape[0])
			disc_pred = disc_pred[rand_idx]
			disc_labels = disc_labels[rand_idx]
			disc_loss = F.binary_cross_entropy_with_logits(disc_pred, disc_labels)
				
	return disc_loss

def prior_loss(args, prior, z, discriminators):
	
	if args.prior_loss == "kl_div":
		z = z.double()
		mu = torch.mean(z, dim=0).float()
		logvar = torch.log(torch.var(z, dim=0).float())
		
		# 0 std cause inf loss so switch value to smallest possible value instead of 0
		# std += torch.finfo(std.dtype).eps
		batch_dist = torch.distributions.normal.Normal(mu, logvar.mul(.5).exp())
		
		unred_loss_prior = torch.distributions.kl.kl_divergence(batch_dist, prior)
		
		loss_prior = unred_loss_prior.sum(1).mean()
		# if torch.isinf(loss_prior):
		# 	with torch.no_grad():
		# 		pdb.set_trace()
		# 		idx = torch.nonzero(torch.isinf(unred_loss_prior))
		# 		print(f"idx:{idx}")
		# 		print(f"mean idx:{mu[idx]}")
		# 		print(f"std idx:{std[idx]}")
		# 		print(f"prior:{unred_loss_prior[idx]}")

		# 		print(f"mean:{torch.mean(mu)}")
		# 		print(f"std:{torch.mean(std)}")

	elif args.prior_loss == "gan":

		prior_disc = discriminators["prior_disc"][0]
		prior_labels = torch.ones(z.shape[0], device=args.device) * REAL_LABEL
		# pdb.set_trace()
		prior_pred = torch.squeeze(prior_disc(z.to(args.device)))
		loss_prior = F.binary_cross_entropy_with_logits(prior_pred, prior_labels)

	return loss_prior

def prior_disc_loss(args, prior, z, discriminators):

	# print(f"prior:{prior}")
	# print(f"z:{z.shape}")
	# print(f"z[0]:{z.shape[0]}")
	prior_z = prior.rsample([z.shape[0]])
	z = torch.cat([z, prior_z])
	prior_labels = torch.cat([torch.ones(prior_z.shape[0]) * FAKE_LABEL,  torch.ones(prior_z.shape[0]) * REAL_LABEL])
	idx = torch.randperm(z.shape[0])
	z = z[idx]
	prior_labels = prior_labels[idx].to(args.device)

	prior_disc = discriminators["prior_disc"][0]
	prior_pred = torch.squeeze(prior_disc(z))
	prior_disc_loss = F.binary_cross_entropy_with_logits(prior_pred, prior_labels)

	return prior_disc_loss

def perturb_loss(z, perturb_labels, args, discriminators):
	"""
		uses latent embeddings and uses discriminators to extract
		features to predict perturb_labels.
		The same function is used for training both the encoder as well
		as the perturbation discriminators.

		returns a dict of losses for perturbation type
	"""
	perturb_loss = dict()
	# pdb.set_trace()
	for perturb_type in perturb_labels:
		# pdb.set_trace()
		perturb_pred = discriminators[f"{perturb_type}_disc"][0](z)

		num_class = perturb_dict[perturb_type].num_class

		if num_class == 2:
			perturb_labels[perturb_type] = perturb_labels[perturb_type].unsqueeze(dim=-1) 
			iter_perturb_loss = F.binary_cross_entropy_with_logits(perturb_pred, perturb_labels[perturb_type])
		else:
			iter_perturb_loss = F.cross_entropy(perturb_pred, perturb_labels[perturb_type].long())
		
		perturb_loss[perturb_type] = iter_perturb_loss

	return perturb_loss

def swap_halves(batch):
	if batch.shape[0]%2 == 0:
		a, b = batch.split(batch.shape[0]//2)
		return torch.cat([b, a])
	else:
		a, b, c = batch.split(batch.shape[0]//2)
		return torch.cat([c, b, a])

def latent_interp(model, z, device):
	"""
		returns images generated from interpolated latent and interpolating factor
	"""
	alpha = torch.rand(z.shape[0], 1, 1, 1).to(device)
	z_interp = torch.lerp(z, swap_halves(z), alpha)
	x_tilde_interp = model.decoder_forward(z_interp)
	
	return x_tilde_interp, alpha

def interp_loss(interp_disc, x_tilde_interp):
	loss_perc = interp_disc(x_tilde_interp)
	loss_perc = torch.mean(loss_perc**2)

	return loss_perc

def interp_disc_loss(interp_disc, pred_alpha, alpha, images, x_tilde, gamma):

	x_interp = torch.lerp(x_tilde, images, gamma)
	loss_alpha = F.mse_loss(pred_alpha, alpha.squeeze())
	loss_reg = interp_disc(x_interp)
	loss_reg = torch.mean(loss_reg**2)

	loss_disc = loss_alpha + loss_reg

	return loss_disc


def test(get_losses, model, test_loader, args, discriminators, testing=False):
	model.eval()

	for disc in discriminators:
		if discriminators[disc][0] is not None and discriminators[disc][1] is None:
			print(f"define an optimizer for discriminator:{disc}.. exiting", file=sys.stderr)
			sys.stderr.flush()
		else:
			discriminators[disc][0].eval()

	with torch.no_grad():
		loss_dict = {}

		if testing:
			b_idx = 0
		total_el = 0
		for data in test_loader:
			
			images = data[0]
			if testing and b_idx >= 50:
				break
			batch_size = images.shape[0]
			images = images.to(args.device)
			if len(loss_dict) == 0:
				loss_dict, z = get_losses(model, images, args, discriminators)
				
				# mu = latent_metrics_dict["mean"]
				# std = latent_metrics_dict["std"]
			
			else:
				iter_losses, iter_z = get_losses(model, images, args, discriminators)
				
				# iter_mu = latent_metrics_dict["mean"]
				# iter_std = latent_metrics_dict["std"]
				
				for loss_type in loss_dict:
					loss_dict[loss_type] += iter_losses[loss_type]
				
				z = torch.cat([z, iter_z])
				
				# n = total_el 
				# m = batch_size
				# mu = (n * mu + m*iter_mu) / (n + m)
				# std = ( ((n-1) * (std**2)) + ((m - 1) * (iter_std**2)) / (n+m-1) ) + ( (n*m*(mu*iter_mu)**2) / ((n+m)*(n+m-1)) )

			total_el += batch_size

			if testing:
				b_idx += 1

		for loss in loss_dict:
			loss_dict[loss] = torch.mean(loss_dict[loss])

	return loss_dict, z


def save_state(model, model_optimizer, discriminators, loss, best_loss, recons_loss_type, step, save_path):

	save_disc_dict = {}

	# save disc weights and resp optim
	for disc in discriminators:
		save_disc_dict[disc] = (discriminators[disc][0].state_dict(), discriminators[disc][1].state_dict())

	save_dict = {"epoch":step,
				 "model":model.state_dict(),
				 "optim":model_optimizer.state_dict(),
				 "discriminators":save_disc_dict}
	if loss < best_loss:
		print("best performance!", file=sys.stderr)
		save_path = os.path.join(save_path, "best.pt")
	else:
		save_path = os.path.join(save_path, f"epoch_{step}.pt")

	print(f"saving model at epoch:{step} in location:{save_path}", file=sys.stderr)
	torch.save(save_dict, save_path)

def load_state(save_path, model, model_optimizer, discriminators):

	ckpt = torch.load(os.path.join(save_path, "best.pt"))
	start_epoch = ckpt["epoch"]
	model.load_state_dict(ckpt["model"])
	
	for disc in discriminators:
		discriminators[disc][0].load_state_dict(ckpt["discriminators"][disc][0])
		discriminators[disc][1].load_state_dict(ckpt["discriminators"][disc][1])

	return start_epoch

def load_model(save_path, model):

	ckpt = torch.load(os.path.join(save_path, "best.pt"))
	model.load_state_dict(ckpt["model"])
