
import numpy as np
import time
import os
import argparse
import sys
import pdb

import torch
import torchvision
import torch.nn.functional as F
from torch.distributions.normal import Normal

from torchvision import datasets, transforms
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

from modules import ACAI, Discriminator, AnchorComparator, ClubbedPermutationComparator, FullPermutationComparator
import train_util

def get_losses(model, images, args, discriminators):
	
	x_tilde, z = model(images)
		
	############1111 AE loss 11111############
	x_tilde_interp, alpha = train_util.latent_interp(model, z, args.device)

	assert x_tilde_interp.shape[2] == images.shape[2] and x_tilde_interp.shape[3] == images.shape[3]

	# Reconstruction loss
	recons_loss = train_util.get_recons_loss(images, x_tilde, args.device, args.recons_loss, discriminators)

	# perceptual/discriminator loss
	#pdb.set_trace()
	loss_perc = train_util.interp_loss(discriminators["interp_disc"][0], x_tilde_interp)

	loss_ae = recons_loss + args.lam * loss_perc

	###########111111 ----------- 111111#########

	###########222222 critic loss 222222#########

	pred_alpha = torch.squeeze(discriminators["interp_disc"][0](x_tilde_interp.detach()))

	# interpolation of recons and orig in data space
	loss_interp_disc = train_util.interp_disc_loss(discriminators["interp_disc"][0], pred_alpha, alpha, images, x_tilde, args.gamma)

	##########222222 ----------- 222222##########

	##########333333 Reconstruction critic loss 333333##########
	if args.recons_loss != "mse":
		loss_recons_disc = train_util.get_recons_disc_loss(args.recons_loss, discriminators["recons_disc"][0], images, x_tilde.detach(), args.device)
	else:
		loss_recons_disc = 0

	##########333333 ----------- 333333##########
	loss_dict = {"recons_loss":loss_ae,
				 "interp_disc_loss":loss_interp_disc}
	if args.recons_loss != "mse":
		loss_dict["recons_disc_loss"] = loss_recons_disc
	
	return loss_dict, z.detach()

def train(model, opt, train_loader, args, discriminators, writer):

	model.train()

	for disc in discriminators:
		if discriminators[disc][0] is not None and discriminators[disc][1] is None:
			print(f"define an optimizer for discriminator:{disc}.. exiting", file=sys.stderr)
			sys.stderr.flush()
		else:
			discriminators[disc][0].train()

	for idx, data in enumerate(train_loader):

		# if there are any discriminators used for training then, graph may have to be retained
		if len(discriminators) > 0:
			retain_graph = True
		else:
			retain_graph = False

		images = data[0]
		images = images.to(args.device)

		loss_dict, z = get_losses(model, images, args, discriminators)
		# print(loss_dict, file=sys.stderr)

		############1111 AE loss 11111############
		opt.zero_grad()
		loss_dict["recons_loss"].backward(retain_graph=retain_graph)
		opt.step()
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
		if idx % 1000 == 0:
			print("iter:", str(args.steps)+"/"+str(args.num_epochs*(len(train_loader))), "iter loss:", loss_dict["recons_loss"].item())
			for disc in discriminators:
				print(f"{disc} loss:{loss_dict[f'{disc}_loss']}")
			sys.stderr.flush()
			sys.stderr.flush()
		
		train_util.log_losses("train", loss_dict, args.steps, writer)

		train_util.log_latent_metrics("train", z, args.steps, writer)
	
def main(args):
	
	writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
	save_filename = './models/{0}'.format(args.output_folder)

	train_loader, val_loader, test_loader = train_util.get_dataloaders(args)
	recons_input_img = train_util.log_input_img_grid(test_loader, writer)

	input_dim = 3
	model = ACAI(args.img_res, input_dim, args.hidden_size, args.enc_type, args.dec_type).to(args.device)
	disc = Discriminator(input_dim, args.img_res, args.input_type).to(args.device)
	disc_opt = torch.optim.Adam(disc.parameters(), lr=args.disc_lr, amsgrad=True)
	# if torch.cuda.device_count() > 1 and args.device == "cuda":
	# 	model = torch.nn.DataParallel(model)
	
	opt = torch.optim.Adam(model.parameters(), lr=args.lr)
	
	# ae_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)

	# interp_disc_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(disc_opt, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)
	
	discriminators = {"interp_disc":[disc, disc_opt]}
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

		discriminators["recons_disc"] = [recons_disc, recons_disc_opt]
	
	# Generate the samples first once
	train_util.save_recons_img_grid("val", recons_input_img, model, 0, args)
	
	if args.weights == "load":
		start_epoch = train_util.load_state(save_filename, model, opt, discriminators)
	else:
		start_epoch = 0

	best_loss = torch.tensor(np.inf)
	for epoch in range(args.num_epochs):
		print("Epoch {}:".format(epoch))
		train(model, opt, train_loader, args, discriminators, writer)
		
		# curr_loss = val(model, val_loader)
		# print(f"epoch val loss:{curr_loss}")

		val_loss_dict, z = train_util.test(get_losses, model, val_loader, args, discriminators)
		
		train_util.log_losses("val", val_loss_dict, epoch+1, writer)
		train_util.log_latent_metrics("val", z, epoch+1, writer)
		
		# train_util.log_recons_img_grid(recons_input_img, model, epoch+1, args.device, writer)
		# train_util.log_interp_img_grid(recons_input_img, model, epoch+1, args.device, writer)

		train_util.save_recons_img_grid("test", recons_input_img, model, epoch+1, args)
		train_util.save_interp_img_grid("test", recons_input_img, model, epoch+1, args)
		
		train_util.save_state(model, opt, discriminators, val_loss_dict["recons_loss"], best_loss, args.recons_loss, epoch, save_filename)

if __name__ == '__main__':

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
	parser.add_argument("--recons_loss", type=str, 
		choices={"mse", "gan", "comp", "comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc"},
		help="type of reconstruction loss to use, discriminator n/w of GAN or mean square error")

	# Optimization
	parser.add_argument('--batch_size', type=int, default=64,
		help='batch size (default: 128)')
	parser.add_argument('--num_epochs', type=int, default=20,
		help='number of epochs (default: 100)')
	parser.add_argument('--lr', type=float, default=2e-4,
		help='learning rate for Adam optimizer (default: 2e-4)')
	parser.add_argument("--disc_lr", type=float, default=None)
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

	args = parser.parse_args()

	if args.disc_lr is None:
		args.disc_lr = args.lr

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

	#print(args.output_folder)
	model_name = f"acai_{args.recons_loss}"
	if args.output_folder is None:
		args.output_folder = os.path.join(model_name, args.dataset, f"depth_{args.enc_type}_{args.dec_type}_hs_{args.img_res}_{args.hidden_size}")

	if not os.path.exists('./models/{0}'.format(args.output_folder)):
		os.makedirs('./models/{0}'.format(args.output_folder))
	args.steps = 0

	print("training acai with following params", file=sys.stderr)
	print(f"batch size: {args.batch_size}", file=sys.stderr)
	print(f"encoder:{args.enc_type}", file=sys.stderr)
	print(f"decoder:{args.dec_type}", file=sys.stderr)
	print(f"loss:{args.recons_loss}", file=sys.stderr)
	print(f"mode:{args.weights}")
	sys.stderr.flush()
	main(args)