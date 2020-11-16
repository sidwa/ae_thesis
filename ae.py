"""
	code modified from https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0
"""
import pdb
import numpy as np
import traceback
import subprocess

import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
from torch.distributions.normal import Normal

from modules import PCIE, to_scalar, Discriminator, AnchorComparator, ClubbedPermutationComparator, FullPermutationComparator, \
					FullPermutationColorComparator, Latent2ClassDiscriminator, LatentMultiClassDiscriminator
import train_util

from tensorboardX import SummaryWriter

from PIL import Image
import time
from tqdm import tqdm
import sys


def get_losses(model, data, args, discriminators):
	images = data[1].to(args.device)
	orig_images = data[0].to(args.device)
	
	perturb_labels = {}
	normalized_perturb_labels = {}
	for key in data[2]:
		# print(f"pert label:{data[2]}")
		# print(f"pert label:{data[2].shape}")
		# pdb.set_trace()
		perturb_labels[key] = data[2][key].to(args.device)
		normalized_perturb_labels[key] = data[3][key].to(args.device)
	
	# print(f"perturb labels:{perturb_labels}")
	x_tilde, z = model(images, normalized_perturb_labels)
	# pdb.set_trace()
	# exclude batch dim from prior shape
	prior_shape = z.shape[1:]
	prior = Normal(torch.zeros(prior_shape, device=args.device), torch.ones(prior_shape, device=args.device))


	loss_dict = dict()
	############1111 AE loss 11111############

	prior_z = prior.rsample([z.shape[0]])
	# print(f"z:{z.shape} prior z:{prior_z.shape}")
	interp_z = torch.cat([z, prior_z], dim=0)
	interp_z = interp_z[torch.randperm(z.shape[0])]

	x_tilde_interp, alpha = train_util.latent_interp(model, interp_z, args.device)

	prior_z = prior.rsample([z.shape[0]])
	prior_gen = model.decoder_forward(prior_z)
	x_tilde_interp = torch.cat([x_tilde_interp, prior_gen])
	
	# 0.5 means perfect half interpolation which in this context is to say
	# that the interpolation discriminator should classify this as a fake
	# since the latent was not taken form an actual datapoint
	prior_gen_alpha = torch.full_like(alpha, 0.5)
	alpha = torch.cat([alpha, prior_gen_alpha])
	assert x_tilde_interp.shape[2] == images.shape[2] and x_tilde_interp.shape[3] == images.shape[3]

	# Reconstruction loss
	recons_loss = train_util.get_recons_loss(orig_images, x_tilde, args.device, args.recons_loss, discriminators)

	# perceptual/discriminator loss
	#pdb.set_trace()
	loss_perc = train_util.interp_loss(discriminators["interp_disc"][0], x_tilde_interp)

	loss_ae = recons_loss + args.lam * loss_perc

	###########111111 ----------- 111111#########

	########### prior loss  #########

	prior_loss = train_util.prior_loss(args, prior, z, discriminators)

	if not torch.isnan(prior_loss) and not torch.isinf(prior_loss):
		loss_ae += prior_loss
	else:
		pass
		# print(f"prior loss:{prior_loss}")
	
	#############################################

	########## perturbation invariance loss ###############
	if args.perturb_feat_gan:
		perturb_loss = train_util.perturb_loss(z, perturb_labels, args, discriminators)

		for perturb_type in perturb_loss:

			# higher perturb loss means latents are perturbation invariant so we subtract this loss
			# from total autoencoder loss
			loss_ae -= perturb_loss[perturb_type]
		
			loss_dict[f"{perturb_type}_disc_loss"] = perturb_loss[perturb_type]
	######################################################

	loss_dict["recons_loss"] = loss_ae
	###########222222 critic loss 222222#########

	pred_alpha = torch.squeeze(discriminators["interp_disc"][0](x_tilde_interp.detach()))

	# interpolation of recons and orig in data space
	loss_interp_disc = train_util.interp_disc_loss(discriminators["interp_disc"][0], pred_alpha, alpha, images, x_tilde, args.gamma)
	loss_dict["interp_disc_loss"] = loss_interp_disc

	if args.prior_loss == "gan":
		loss_prior_disc = train_util.prior_disc_loss(args, prior, z, discriminators)
		loss_dict["prior_disc_loss"] = loss_prior_disc

	##########222222 ----------- 222222##########

	##########333333 Reconstruction critic loss 333333##########
	if args.recons_loss != "mse":
		loss_recons_disc = train_util.get_recons_disc_loss(args.recons_loss, discriminators["recons_disc"][0], images, x_tilde.detach(), args.device)
		loss_dict["recons_disc_loss"] = loss_recons_disc

	##########333333 ----------- 333333##########
	# print(f"recons loss:{recons_loss.item()} \t perc loss:{loss_perc.item()} \t prior loss:{prior_loss.item()}")
	return loss_dict, z.detach()

def train(epoch, data_loader, model, optimizer, args, writer, discriminators):
	# use a dict or add interp dict and optim separately
	model.train()

	for disc in discriminators:
		if discriminators[disc][0] is not None and discriminators[disc][1] is None:
			print(f"define an optimizer for discriminator:{disc}.. exiting", file=sys.stderr)
			sys.stderr.flush()
		else:
			discriminators[disc][0].train()

	for idx, data in enumerate(data_loader):
		retain_graph = True
		
		# print(f"shape:{data.shape}")
		loss_dict, z = get_losses(model, data, args, discriminators)


		############1111 AE loss 11111############
		optimizer.zero_grad()
		loss_dict["recons_loss"].backward(retain_graph=retain_graph)
		optimizer.step()
		###########111111 ----------- 111111#########
		
		# backprop all discriminators
		for disc_idx, disc in enumerate(discriminators):
			discriminators[disc][1].zero_grad()

			if disc_idx >= len(discriminators)-1:
				retain_graph = False
			
			#print(f"disc:{disc}\t loss:{loss_dict[f'{disc}_loss']}\t retain:{retain_graph}", file=sys.stderr)
			loss_dict[f"{disc}_loss"].backward(retain_graph=retain_graph)
			discriminators[disc][1].step()
		
		# Logs
		if idx % args.print_interval == 0:
			print("iter:", str(args.steps)+"/"+str(args.num_epochs*(len(data_loader))), "iter loss:", loss_dict["recons_loss"].item(), 
				file=sys.stderr)
			print(f"memory:\n{subprocess.run('nvidia-smi')}")
			for disc in discriminators:
				print(f"{disc} loss:{loss_dict[f'{disc}_loss']}", file=sys.stderr)
			
			sys.stderr.flush()
			sys.stderr.flush()
		
		train_util.log_losses("train", loss_dict, args.steps, writer)

		train_util.log_latent_metrics("train", z, args.steps, writer)

		if idx == 0:
			print(torch.cuda.max_memory_allocated(torch.device(args.device)), file=sys.stderr)
			sys.stderr.flush()

		args.steps += 1


def main(args):
	writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
	save_filename = './models/{0}'.format(args.output_folder)


	d_args = vars(args)
	pert_types = train_util.get_perturb_types(args)
	train_loader, valid_loader, test_loader = train_util.get_dataloaders(args, pert_types)
	recons_input_img = train_util.log_input_img_grid(test_loader, writer)
	

	# print(f"nn num:{len(pert_types)}")
	num_perturb_types = len(pert_types)

	input_dim = 3
	model = PCIE(args.img_res, input_dim, args.hidden_size, num_perturb_types, args.enc_type, args.dec_type)
	interp_disc = Discriminator(input_dim, args.img_res, args.input_type).to(args.device)
	interp_disc_opt = torch.optim.Adam(interp_disc.parameters(), lr=args.disc_lr, amsgrad=True)
	# if torch.cuda.device_count() > 1 and args.device == "cuda":
	# 	model = torch.nn.DataParallel(model)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	
	# ae_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)

	# interp_disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(interp_disc_opt, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)
	
	discriminators = {"interp_disc":[interp_disc, interp_disc_opt]}

	if args.recons_loss != "mse":
		if args.recons_loss == "gan":
			recons_disc = Discriminator(input_dim, args.img_res, args.input_type).to(args.device)
		elif args.recons_loss == "comp":
			recons_disc = AnchorComparator(input_dim*2, args.img_res, args.input_type).to(args.device)
		elif "comp_2" in args.recons_loss:
			recons_disc = ClubbedPermutationComparator(input_dim*2, args.img_res, args.input_type).to(args.device)
		elif "comp_6" in args.recons_loss:
			if "color" in args.recons_loss:
				recons_disc = FullPermutationColorComparator(input_dim*2, args.img_res, args.input_type).to(args.device)
			else:
				recons_disc = FullPermutationComparator(input_dim*2, args.img_res, args.input_type).to(args.device)

		recons_disc_opt = torch.optim.Adam(recons_disc.parameters(), lr=args.disc_lr, amsgrad=True)
		recons_disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(recons_disc_opt, "min", patience=args.lr_patience, factor=0.5, 
		threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)

		discriminators["recons_disc"] = [recons_disc, recons_disc_opt]
		
	if args.prior_loss == "gan":
		prior_disc = Latent2ClassDiscriminator(args.hidden_size, args.img_res // args.scale_factor)
		prior_disc_opt = torch.optim.Adam(prior_disc.parameters(), lr=args.disc_lr, amsgrad=True)
		# prior_disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(prior_disc_opt, "min", patience=args.lr_patience, factor=0.5, 
		# threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)

		discriminators["prior_disc"] = [prior_disc, prior_disc_opt]


	if args.perturb_feat_gan:
		for perturb_type in train_util.perturb_dict:
			if d_args[perturb_type]:
				num_class = train_util.perturb_dict[perturb_type].num_class
				if num_class == 2:
					pert_disc = Latent2ClassDiscriminator(args.hidden_size, args.img_res // args.scale_factor)
					pert_disc_opt = torch.optim.Adam(pert_disc.parameters(), lr=args.disc_lr, amsgrad=True)
				else:
					pert_disc = LatentMultiClassDiscriminator(args.hidden_size, args.img_res // args.scale_factor, num_class)
					pert_disc_opt = torch.optim.Adam(pert_disc.parameters(), lr=args.disc_lr, amsgrad=True)

				discriminators[f"{perturb_type}_disc"] = (pert_disc, pert_disc_opt)

	model.to(args.device)
	for disc in discriminators:
			discriminators[disc][0].to(args.device)

	# Generate the samples first once
	# train_util.log_recons_img_grid(recons_input_img, model, 0, args.device, writer)
	
	if args.weights == "load":
		start_epoch = train_util.load_state(save_filename, model, optimizer, discriminators)
	else:
		start_epoch = 0

	# stop_patience = args.stop_patience
	best_loss = torch.tensor(np.inf)
	for epoch in tqdm(range(start_epoch, args.num_epochs), file=sys.stdout):

		# for CUDA OOM error, prevents running dependency job on slurm which is meant to run on timeout
		try:
			train(epoch, train_loader, model, optimizer, args, writer, discriminators)
			# pass
		except RuntimeError as err:
			print("".join(traceback.TracebackException.from_exception(err).format()), file=sys.stderr)
			print("*******", file=sys.stderr)
			print(err, file=sys.stderr)
			exit(0)

		print("out of train")
		# comp = subprocess.run("nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits", text=True, stdout=subprocess.PIPE)
		# print(comp.stdout, file=sys.stderr)
		val_loss_dict, _ = ae_test(get_losses, model, valid_loader, args, discriminators)
		train_util.log_losses("val", val_loss_dict, epoch+1, writer)
		# print("logg loss")
		# train_util.log_latent_metrics("val", z, epoch+1, writer)
		# print("log metric")
		train_util.save_recons_img_grid("test", recons_input_img, model, epoch+1, args)
		# print("log recons")
		train_util.save_interp_img_grid("test", recons_input_img, model, epoch+1, args)
		# print("log interp")
		train_util.save_state(model, optimizer, discriminators, val_loss_dict["recons_loss"], best_loss, args.recons_loss, epoch, save_filename)
		# print("epoch complete and logged")
		#early stop check
		# if val_loss_dict["recons_loss"] - best_loss < args.threshold:
		# 	stop_patience -= 1
		# else:
		# 	stop_patience = args.stop_patience
		
		# if stop_patience == 0:
		# 	print("training early stopped!")
		# 	break

		# ae_lr_scheduler.step(val_loss_dict["recons_loss"])
		# interp_disc_lr_scheduler.step(val_loss_dict["interp_disc_loss"])
		# if args.recons_loss != "mse":
		# 	recons_disc_lr_scheduler.step(val_loss_dict["recons_disc_loss"])

def ae_test(get_losses, model, test_loader, args, discriminators, testing=False):
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
			batch_size = data[0].shape[0]
			if len(loss_dict) == 0:
				loss_dict, z = get_losses(model, data, args, discriminators)
				
				# mu = latent_metrics_dict["mean"]
				# std = latent_metrics_dict["std"]
			
			else:
				iter_losses, iter_z = get_losses(model, data, args, discriminators)
				
				# iter_mu = latent_metrics_dict["mean"]
				# iter_std = latent_metrics_dict["std"]
				
				for loss_type in loss_dict:
					loss_dict[loss_type] += iter_losses[loss_type]
				
				# z = torch.cat([z, iter_z])
				
				# n = total_el 
				# m = batch_size
				# mu = (n * mu + m*iter_mu) / (n + m)
				# std = ( ((n-1) * (std**2)) + ((m - 1) * (iter_std**2)) / (n+m-1) ) + ( (n*m*(mu*iter_mu)**2) / ((n+m)*(n+m-1)) )

			total_el += batch_size

			if testing:
				b_idx += 1

		for loss in loss_dict:
			loss_dict[loss] = torch.mean(loss_dict[loss])

	return loss_dict, z.detach()

def val_test(args):
	writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
	save_filename = './models/{0}'.format(args.output_folder)

	d_args = vars(args)

	num_perturb_types = 0
	perturb_types = []
	for perturb_type in train_util.perturb_dict:
		if d_args[perturb_type]:
			num_perturb_types += 1
			perturb_types.append(perturb_type)

	train_loader, valid_loader, test_loader = train_util.get_dataloaders(args, perturb_types)

	recons_input_img = train_util.log_input_img_grid(test_loader, writer)

	input_dim = 3
	model = PCIE(args.img_res, input_dim, args.hidden_size, num_perturb_types, args.enc_type, args.dec_type)
	interp_disc = Discriminator(input_dim, args.img_res, args.input_type).to(args.device)
	interp_disc_opt = torch.optim.Adam(interp_disc.parameters(), lr=args.disc_lr, amsgrad=True)
	# if torch.cuda.device_count() > 1 and args.device == "cuda":
	# 	model = torch.nn.DataParallel(model)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	
	# ae_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)

	# interp_disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(interp_disc_opt, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)
	
	discriminators = {"interp_disc":[interp_disc, interp_disc_opt]}

	if args.recons_loss != "mse":
		if args.recons_loss == "gan":
			recons_disc = Discriminator(input_dim, args.img_res, args.input_type).to(args.device)
		elif args.recons_loss == "comp":
			recons_disc = AnchorComparator(input_dim*2, args.img_res, args.input_type).to(args.device)
		elif "comp_2" in args.recons_loss:
			recons_disc = ClubbedPermutationComparator(input_dim*2, args.img_res, args.input_type).to(args.device)
		elif "comp_6" in args.recons_loss:
			recons_disc = FullPermutationComparator(input_dim*2, args.img_res, args.input_type).to(args.device)

		recons_disc_opt = torch.optim.Adam(recons_disc.parameters(), lr=args.disc_lr, amsgrad=True)
		recons_disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(recons_disc_opt, "min", patience=args.lr_patience, factor=0.5, 
		threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)

		discriminators["recons_disc"] = [recons_disc, recons_disc_opt]
		
	if args.prior_loss == "gan":
		prior_disc = Latent2ClassDiscriminator(args.hidden_size, args.img_res // args.scale_factor)
		prior_disc_opt = torch.optim.Adam(prior_disc.parameters(), lr=args.disc_lr, amsgrad=True)
		# prior_disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(prior_disc_opt, "min", patience=args.lr_patience, factor=0.5, 
		# threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)

		discriminators["prior_disc"] = [prior_disc, prior_disc_opt]


	print("pertrub gans")

	if args.perturb_feat_gan:
		for perturb_type in train_util.perturb_dict:
			if d_args[perturb_type]:
				num_classes = train_util.perturb_dict[perturb_type].num_class
				pdb.set_trace()
				if num_classes == 2:
					
					print(f"perturb:{d_args[perturb_type]}\ttype: two ")
					pert_disc = Latent2ClassDiscriminator(args.hidden_size, args.img_res // args.scale_factor)
					pert_disc_opt = torch.optim.Adam(pert_disc.parameters(), lr=args.disc_lr, amsgrad=True)
				else:
					print(f"perturb:{d_args[perturb_type]}\ttype: multi ")
					pert_disc = LatentMultiClassDiscriminator(args.hidden_size, args.img_res // args.scale_factor, 
																num_classes)
					pert_disc_opt = torch.optim.Adam(pert_disc.parameters(), lr=args.disc_lr, amsgrad=True)

				
				discriminators[f"{perturb_type}_disc"] = (pert_disc, pert_disc_opt)

	print("perrturb gans set")

	model.to(args.device)
	for disc in discriminators:
			discriminators[disc][0].to(args.device)

	# Generate the samples first once
	# train_util.log_recons_img_grid(recons_input_img, model, 0, args.device, writer)
	
	if args.weights == "load":
		start_epoch = train_util.load_state(save_filename, model, optimizer, discriminators)
	else:
		start_epoch = 0

	# stop_patience = args.stop_patience
	best_loss = torch.tensor(np.inf)
	for epoch in tqdm(range(start_epoch, 4), file=sys.stdout):
		val_loss_dict = train_util.test(get_losses, model, valid_loader, args, discriminators, True)
		if args.weights == "init" and epoch==1:
			epoch+=1
			break

	train_util.log_losses("val", val_loss_dict, epoch+1, writer)
	train_util.log_recons_img_grid(recons_input_img, model, epoch, args.device, writer)
	train_util.log_interp_img_grid(recons_input_img, model, epoch+1, args.device, writer)
		
	train_util.save_state(model, optimizer, discriminators, val_loss_dict["recons_loss"], best_loss, args.recons_loss, epoch, save_filename)


	print(val_loss_dict)

if __name__ == '__main__':
	import argparse
	import os
	import multiprocessing as mp

	start_time = time.time()

	parser = argparse.ArgumentParser(description='ACAI')

	# General
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
		choices={"mse", "gan", "comp", "comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc", "comp_6_adv_color", "comp_6_dc_color"},
		help="type of reconstruction loss to use, discriminator n/w of GAN or mean square error")
	parser.add_argument("--prior_loss", type=str,
		choices={"kl_div", "gan"}, default="kl_div",
		help="force latents to fall in a prior distribution using this loss")

	
	# perturb factors

	# perturbation types  
	for perturb_type in train_util.perturb_dict:
		parser.add_argument(f"--{perturb_type}", action="store_true",
		help=train_util.perturb_dict[perturb_type].help)

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
	parser.add_argument('--print_interval', type=int, default=500,
		help='number of workers for trajectories sampling (default: 0)')
	parser.add_argument('--device', type=str, default='cuda', choices={"cpu", "cuda", "cuda:0", "cuda:1"},
		help='set the device (cpu or cuda, default: cpu)')

	args = parser.parse_args()

	# Create logs and models folder if they don't exist
	if not os.path.exists('./logs'):
		os.makedirs('./logs')
	if not os.path.exists('./models'):
		os.makedirs('./models')
	# Device
	#print("chosen", args.device)
	args.device = torch.device(args.device
		if torch.cuda.is_available() else 'cpu')
	print("set", args.device, file=sys.stderr)
	sys.stderr.flush()
	# Slurm


	perturb_args = train_util.get_perturb_types(args)

	#print(args.output_folder)
	model_name = f"ae_{args.recons_loss}"
	if args.output_folder is None:
		args.output_folder = os.path.join(
			model_name, 
			args.dataset,
			f"prior_loss_{args.prior_loss}_pert_feat_{perturb_args}",
			f"pert_gan_{args.perturb_feat_gan}",
			f"depth_{args.enc_type}_{args.dec_type}_hs_{args.img_res}_{args.hidden_size}")

	if not os.path.exists('./models/{0}'.format(args.output_folder)):
		os.makedirs('./models/{0}'.format(args.output_folder))
	args.steps = 0

	print("training ae with following params", file=sys.stderr)
	print(f"batch size: {args.batch_size}", file=sys.stderr)
	print(f"encoder:{args.enc_type}", file=sys.stderr)
	print(f"decoder:{args.dec_type}", file=sys.stderr)
	print(f"loss:{args.recons_loss}", file=sys.stderr)
	print(f"mode:{args.weights}")
	print(f"save:{args.output_folder}")
	sys.stderr.flush()
	# exit()
	main(args)

	# val_test(args)