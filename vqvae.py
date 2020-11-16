from tqdm import tqdm
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import VectorQuantizedVAE, to_scalar, Discriminator, AnchorComparator, ClubbedPermutationComparator, FullPermutationComparator
from datasets import MiniImagenet

import train_util

from tensorboardX import SummaryWriter

from PIL import Image
import time
import sys


def get_losses(model, images, args, discriminators):
	images = images.to(args.device)

	x_tilde, z_q_x, z_e_x = model(images)

	# Reconstruction loss
	recons_loss = train_util.get_recons_loss(images, x_tilde, args.device, args.recons_loss, discriminators)

	# Vector quantization objective
	loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
		
	# Commitment objective
	loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

	loss = recons_loss + loss_vq + args.beta * loss_commit

	if args.recons_loss != "mse":
		loss_recons_disc = train_util.get_recons_disc_loss(args.recons_loss, discriminators["recons_disc"][0], images, x_tilde.detach(), args.device)


	loss_dict = {"recons_loss":loss,
				 "loss_quant":(loss_vq + args.beta * loss_commit)}
	if args.recons_loss != "mse":
		loss_dict["recons_disc_loss"] = loss_recons_disc

	return loss_dict, z_q_x.detach()

def train(epoch, data_loader, model, optimizer, args, writer, discriminators):

	for disc in discriminators:
		if discriminators[disc][0] is not None and discriminators[disc][1] is None:
			print(f"define an optimizer for discriminator:{disc}.. exiting", file=sys.stderr)
		else:
			discriminators[disc][0].train()

	torch.cuda.empty_cache()
	
	for idx, data in enumerate(data_loader):

		# if there are any discriminators used for training then, graph may have to be retained
		if len(discriminators) > 0:
			retain_graph = True
		else:
			retain_graph = False

		images = data[0]
		images = images.to(args.device)

		loss_dict, z = get_losses(model, images, args, discriminators)

		optimizer.zero_grad()
		loss_dict["recons_loss"].backward(retain_graph=retain_graph)
		optimizer.step()

		# backprop all discriminators
		for disc_idx, disc in enumerate(discriminators):
			discriminators[disc][1].zero_grad()
			if disc_idx >= len(discriminators)-1:
				retain_graph = False
			loss_dict[f"{disc}_loss"].backward(retain_graph=retain_graph)
			discriminators[disc][1].step()

		# Logs
		if idx % 1000 == 0:
			print(f"iter:{args.steps}\trecons loss:{loss_dict['recons_loss']}", file=sys.stderr)
			for disc in discriminators:
				print(f"{disc} loss:{loss_dict[f'{disc}_loss']}", file=sys.stderr)
			sys.stderr.flush()
		
		train_util.log_losses("train", loss_dict, args.steps, writer)
		train_util.log_latent_metrics("train", z, args.steps, writer)


		if idx == 0:
			print(torch.cuda.max_memory_allocated(torch.device("cuda:0")), file=sys.stderr)
			#print(torch.cuda.max_memory_allocated(torch.device("cuda:1")), file=sys.stderr)
			sys.stderr.flush()
		
		args.steps += 1

def main(args):
	writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
	save_filename = './models/{0}'.format(args.output_folder)
	
	train_loader, valid_loader, test_loader = train_util.get_dataloaders(args)

	num_channels = 3
	model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k, args.enc_type, args.dec_type)
	model.to(args.device)

	# Fixed images for Tensorboard
	recons_input_img = train_util.log_input_img_grid(test_loader, writer)

	train_util.log_recons_img_grid(recons_input_img, model, 0, args.device, writer)


	discriminators = {}

	input_dim = 3
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
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
	ae_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=args.lr_patience, factor=0.5, 
		threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)
	
	if torch.cuda.device_count() > 1:
		model = train_util.ae_data_parallel(model)
		for disc in discriminators:
			discriminators[disc][0] = torch.nn.DataParallel(discriminators[disc][0])
	
	model.to(args.device)
	for disc in discriminators:
			discriminators[disc][0].to(args.device)

	# Generate the samples first once
	recons_input_img = train_util.log_input_img_grid(test_loader, writer)
	train_util.log_recons_img_grid(recons_input_img, model, 0, args.device, writer)

	if args.weights == "load":
		start_epoch = train_util.load_state(save_filename, model, optimizer, discriminators)
	else:
		start_epoch = 0

	stop_patience = args.stop_patience
	best_loss = torch.tensor(np.inf)
	for epoch in tqdm(range(start_epoch, args.num_epochs), file=sys.stdout):

		try:
			train(epoch, train_loader, model, optimizer, args, writer, discriminators)
		except RuntimeError as err:
			print("".join(traceback.TracebackException.from_exception(err).format()), file=sys.stderr)
			print("*******")
			print(err, file=sys.stderr)
			print(f"batch_size:{args.batch_size}", file=sys.stderr)
			exit(0)

		val_loss_dict, z = train_util.test(get_losses, model, valid_loader, args, discriminators)

		train_util.log_recons_img_grid(recons_input_img, model, epoch+1, args.device, writer)
		train_util.log_interp_img_grid(recons_input_img, model, epoch+1, args.device, writer)

		train_util.log_losses("val", val_loss_dict, epoch+1, writer)
		train_util.log_latent_metrics("val", z, epoch+1, writer)
		train_util.save_state(model, optimizer, discriminators, val_loss_dict["recons_loss"], best_loss, args.recons_loss, epoch, save_filename)

		# early stop check
		# if val_loss_dict["recons_loss"] - best_loss < args.threshold:
		# 	stop_patience -= 1
		# else:
		# 	stop_patience = args.stop_patience
		
		# if stop_patience == 0:
		# 	print("training early stopped!")
		# 	break

		ae_lr_scheduler.step(val_loss_dict["recons_loss"])
		if args.recons_loss != "mse":
			recons_disc_lr_scheduler.step(val_loss_dict["recons_disc_loss"])



def val_test(args):
	writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
	save_filename = './models/{0}'.format(args.output_folder)

	train_loader, valid_loader, test_loader = train_util.get_dataloaders(args)
	recons_input_img = train_util.log_input_img_grid(test_loader, writer)

	input_dim = 3
	model = VectorQuantizedVAE(input_dim, args.hidden_size, args.k, args.enc_type, args.dec_type)
	# if torch.cuda.device_count() > 1 and args.device == "cuda":
	# 	model = torch.nn.DataParallel(model)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
		
	discriminators = {}

	if args.recons_loss == "gan":
		recons_disc = Discriminator(input_dim, args.img_res, args.input_type).to(args.device)
		recons_disc_opt = torch.optim.Adam(recons_disc.parameters(), lr=args.disc_lr, amsgrad=True)
		discriminators["recons_disc"] = [recons_disc, recons_disc_opt]

	model.to(args.device)
	for disc in discriminators:
			discriminators[disc][0].to(args.device)
	

	if args.weights == "load":
		start_epoch = train_util.load_state(save_filename, model, optimizer, discriminators)
	else:
		start_epoch = 0

	stop_patience = args.stop_patience
	best_loss = torch.tensor(np.inf)
	for epoch in tqdm(range(start_epoch, 4), file=sys.stdout):
		val_loss_dict, z = train_util.test(get_losses, model, valid_loader, args, discriminators, True)
		# if args.weights == "init" and epoch==1:
		# 	epoch+=1
		# 	break


		train_util.log_recons_img_grid(recons_input_img, model, epoch+1, args.device, writer)
		train_util.log_interp_img_grid(recons_input_img, model, epoch+1, args.device, writer)

		train_util.log_losses("val", val_loss_dict, epoch+1, writer)
		train_util.log_latent_metrics("val", z, epoch+1, writer)
		train_util.save_state(model, optimizer, discriminators, val_loss_dict["recons_loss"], best_loss, args.recons_loss, epoch, save_filename)



	print(val_loss_dict)


if __name__ == '__main__':
	import argparse
	import os
	import multiprocessing as mp

	start_time = time.time()

	parser = argparse.ArgumentParser(description='VQ-VAE')

	# General
	parser.add_argument('--data_folder', type=str,
		help='name of the data folder')
	parser.add_argument('--dataset', type=str, choices={"imagenet", "tiny-imagenet"}, default="imagenet",
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
	parser.add_argument("--recons_loss", type=str, 
		choices={"mse", "gan", "comp", "comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc"},
		help="type of reconstruction loss to use, discriminator n/w of GAN or mean square error")

	# Latent space
	parser.add_argument('--hidden_size', type=int, default=256,
		help='size of the latent vectors (default: 256)')
	parser.add_argument('--k', type=int, default=512,
		help='number of latent vectors (default: 512)')
	parser.add_argument('--deep_net', action='store_true',
		help='use deeper version of vqvae which compresses \
		input img by factor of 16 instead of 4 along each dim')

	# Optimization
	parser.add_argument('--batch_size', type=int, default=128,
		help='batch size (default: 128)')
	parser.add_argument('--num_epochs', type=int, default=20,
		help='number of epochs (default: 100)')
	parser.add_argument('--lr', type=float, default=2e-4,
		help='learning rate for Adam optimizer (default: 2e-4)')
	parser.add_argument("--disc_lr", type=float, default=1e-5)
	parser.add_argument('--beta', type=float, default=0.25,
		help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
	parser.add_argument("--threshold", type=float, default=1.0,
					help="val loss reduction threshold")
	parser.add_argument("--lr_patience", type=int, default=2,
					help="epochs of patience for val loss reduction under threshold before reducing learning rate")
	parser.add_argument("--stop_patience", type=int, default=3,
					help="epochs of patience for val loss reduction under threshold before early stop")

	# Miscellaneous
	parser.add_argument('--output_folder', type=str, default='',
		help='name of the output folder (default: vqvae)')
	parser.add_argument('--num_workers', type=int, default=0,
		help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
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


	model_name = f"vqvae_{args.recons_loss}"
	if args.output_folder == "":
		args.output_folder = os.path.join(model_name, args.dataset, f"depth_{args.enc_type}_{args.dec_type}_hs_{args.img_res}_{args.hidden_size}")

	if not os.path.exists('./models/{0}'.format(args.output_folder)):
		os.makedirs('./models/{0}'.format(args.output_folder))
	args.steps = 0

	print("training vqvae with following params", file=sys.stderr)
	print(f"batch size: {args.batch_size}", file=sys.stderr)
	print(f"encoder:{args.enc_type}", file=sys.stderr)
	print(f"decoder:{args.dec_type}", file=sys.stderr)
	print(f"loss:{args.recons_loss}", file=sys.stderr)
	print(f"mode:{args.weights}")

	main(args)

	#val_test(args)