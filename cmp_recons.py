import pdb
import numpy as np
import traceback

import torch
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image, make_grid
from torch.distributions.normal import Normal

from modules import PCIE, to_scalar

import train_util



def main(args):
	save_filename = './models/{0}'.format(args.output_folder)


	d_args = vars(args)
	pert_types = train_util.get_perturb_types(args)
	_, _, test_loader = train_util.get_dataloaders(args, pert_types)
	
	recons_input_img = next(iter(test_loader))[0].to(args.device)[:args.num_img]

	# print(f"nn num:{len(pert_types)}")
	num_perturb_types = len(pert_types)

	input_dim = 3
	model = PCIE(args.img_res, input_dim, args.hidden_size, num_perturb_types, args.enc_type, args.dec_type)
	
	# ae_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)

	# interp_disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(interp_disc_opt, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)
	recons_img = []
	recons_img.append(recons_input_img)

	model.to(args.device)	

	perturb_args = train_util.get_perturb_types(args)

	#print(args.output_folder)
	if args.compare == "recons_loss":
		losses = ["mse", "comp","comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc"]
		prior_loss = args.prior_loss
		perturb_feat_gan = args.perturb_feat_gan

		
		for recons_loss in losses:
			save_dir = os.path.join("./models", compute_save_dir(recons_loss, perturb_feat_gan, 
									perturb_args, prior_loss, args.enc_type, args.dec_type, 
									args.img_res, args.hidden_size))

			train_util.load_model(save_dir, model)

			recons, _ = model(recons_input_img)

			recons_img.append(recons)

		recons_img = torch.cat(recons_img, dim=0)

		fname = '-'.join(losses)
	elif args.compare == "prior_loss":
		recons_loss = args.recons_loss
		perturb_feat_gan = args.perturb_feat_gan
		prior_losses = ["kl_div", "gan"]

		
		for prior_loss in prior_losses:
			save_dir = os.path.join("./models", compute_save_dir(recons_loss, perturb_feat_gan, 
									perturb_args, prior_loss, args.enc_type, args.dec_type, 
									args.img_res, args.hidden_size))

			train_util.load_model(save_dir, model)

			recons, _ = model(recons_input_img)

			recons_img.append(recons)

		recons_img = torch.cat(recons_img, dim=0)

		fname = '-'.join(prior_losses)
	elif args.compare == "perturb_gan":
		recons_loss = args.recons_loss
		prior_loss = args.prior_loss
		perturb_feat_gans = [True, False]

		
		for perturb_feat_gan in perturb_feat_gans:
			save_dir = os.path.join("./models", compute_save_dir(recons_loss, perturb_feat_gan, 
									perturb_args, prior_loss, args.enc_type, args.dec_type, 
									args.img_res, args.hidden_size))

			train_util.load_model(save_dir, model)

			recons, _ = model(recons_input_img)

			recons_img.append(recons)

		recons_img = torch.cat(recons_img, dim=0)

		perturb_feat_gans = [ str(val) for val in perturb_feat_gans ]
		fname = '-'.join(perturb_feat_gans)

	# pdb.set_trace()
	result_dir = "./results"
	result_path = os.path.join("./results", f"cmp_{args.compare}_{ fname }.png")
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)
	torchvision.utils.save_image(
			recons_img,
			result_path,
			nrow=args.num_img
		)


def compute_save_dir(recons_loss, perturb_feat_gan, perturb_args, prior_loss, enc_type="shallow", dec_type="shallow", img_res=64, hidden_size=128):
	model_name = f"ae_{recons_loss}"
	
	output_folder = os.path.join(
		model_name, 
		args.dataset,
		f"prior_loss_{prior_loss}_pert_feat_{perturb_args}",
		f"pert_gan_{perturb_feat_gan}",
		f"depth_{enc_type}_{dec_type}_hs_{img_res}_{hidden_size}")

	if not os.path.exists('./models/{0}'.format(output_folder)):
		print(f"path:{'./models/{0}'.format(output_folder)}")
		raise ValueError(f"model with given params:\nrecons_loss:{recons_loss}\nprior_loss:{prior_loss}\nperturb_gan:{perturb_feat_gan}\n has not been trained")

	return output_folder



if __name__ == '__main__':
	import argparse
	import os
	import multiprocessing as mp
	import sys

	parser = argparse.ArgumentParser(description='PCI-AE')

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
	parser.add_argument('compare', type=str, choices={"recons_loss", "prior_loss", "perturb_gan"})
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

	# perturbation types  
	for perturb_type in train_util.perturb_dict:
		parser.add_argument(f"--{perturb_type}", action="store_true",
		help=train_util.perturb_dict[perturb_type].help)

	parser.add_argument("--perturb_feat_gan", action="store_true",
		help="choose if a discriminator should test if any info of perturbations can be extracted from latents")
	
	# Miscellaneous
	parser.add_argument("--num_img", type=int, default=10,
		help="number of images to reconstruct and compare")
	parser.add_argument('--output_folder', type=str,
		help='name of the output folder')
	parser.add_argument('--num_workers', type=int, default=0,
		help='number of workers for trajectories sampling (default: 0)')
	parser.add_argument('--print_interval', type=int, default=1000,
		help='number of workers for trajectories sampling (default: 0)')
	parser.add_argument('--device', type=str, default='cuda', choices={"cpu", "cuda", "cuda:0", "cuda:1"},
		help='set the device (cpu or cuda, default: cpu)')

	args = parser.parse_args()

	# Device
	#print("chosen", args.device)
	args.device = torch.device(args.device
		if torch.cuda.is_available() else 'cpu')
	# Slurm
		
	args.steps = 0

	args.batch_size = args.num_img

	main(args)