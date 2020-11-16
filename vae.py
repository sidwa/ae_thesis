import numpy as np
from datetime import datetime
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import sys
import traceback

import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.distributions.normal import Normal

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from modules import VAE, Discriminator, AnchorComparator, ClubbedPermutationComparator, FullPermutationComparator
import train_util

import pdb

def get_losses(model, images, args, discriminators):


	x_tilde, z, kl_d = model(images)

	if torch.sum(torch.isnan(x_tilde)) > 0:
		print(f"bad recons! {torch.sum(torch.isnan(x_tilde))} pixels corrupted", file=sys.stderr)
		with torch.no_grad():
			print(f"nan for encode {torch.sum(torch.isnan(model.encoder(images)))}", file=sys.stderr)

	recons_loss = train_util.get_recons_loss(images, x_tilde, args.device, args.recons_loss, discriminators)

	kl_d = torch.mean(kl_d)
	loss = recons_loss + kl_d
	#print(f"total:{loss}\nkl:{kl_d}\nrecons:{recons_loss}")
	# nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(images)
	# log_px = nll.mean().item() - np.log(128) + kl_d.item()
	# log_px /= np.log(2)
		

	if args.recons_loss != "mse":
		loss_recons_disc = train_util.get_recons_disc_loss(args.recons_loss, discriminators["recons_disc"][0], images, x_tilde.detach(), args.device)


	loss_dict = {"recons_loss":loss,
				 "kl_divergence":kl_d}
	if args.recons_loss != "mse":
		loss_dict["recons_disc_loss"] = loss_recons_disc

	return loss_dict, z.detach()

def train(model, train_loader, opt, epoch, writer, args, discriminators):
	model.train()

	for disc in discriminators:
		if discriminators[disc][0] is not None and discriminators[disc][1] is None:
			print(f"define an optimizer for discriminator:{disc}.. exiting", file=sys.stderr)
		else:
			discriminators[disc][0].train()

	for batch_idx, data in enumerate(train_loader):	

		# if there are any discriminators used for training then, graph may have to be retained
		if len(discriminators) > 0:
			retain_graph = True
		else:
			retain_graph = False


		images = data[0]
		images = images.to(args.device)

		loss_dict, z = get_losses(model, images, args, discriminators)
		
		opt.zero_grad()
		print(f"recons_loss retain?:{retain_graph}")
		loss_dict["recons_loss"].backward(retain_graph=retain_graph)
		opt.step()

		# backprop all discriminators
		for disc_idx, disc in enumerate(discriminators):
			discriminators[disc][1].zero_grad()
			if disc_idx >= len(discriminators)-1:
				retain_graph = False
			print(f"disc_loss retain?:{retain_graph}")
			loss_dict[f"{disc}_loss"].backward(retain_graph=retain_graph)
			discriminators[disc][1].step()

		train_util.log_losses("train", loss_dict, args.steps, writer)
		train_util.log_latent_metrics("train", z, args.steps, writer)


		# if (batch_idx + 1) % 1000 == 0:
		# 	print(f"iter:{args.steps}\trecons loss:{loss_dict['recons_loss']}", file=sys.stderr)
		# 	for disc in discriminators:
		# 		print(f"{disc} loss:{loss_dict[f'{disc}_loss']}", file=sys.stderr)
		# 	sys.stderr.flush()
	
		# if batch_idx == 0:
		# 	print(torch.cuda.max_memory_allocated(torch.device(args.device)), file=sys.stderr)
		# 	sys.stderr.flush()
		
		args.steps += 1

def main(args):
	
	
	input_dim=3
	model = VAE(input_dim, args.hidden_size, args.enc_type, args.dec_type)

	opt = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
	# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=args.lr_patience, factor=0.5, 
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-6)


	# ae_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=args.lr_patience, factor=0.5,
	# 	threshold=args.threshold, threshold_mode="abs", min_lr=1e-7)
	
	discriminators = {}

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

	if torch.cuda.device_count() > 1:
		model = train_util.ae_data_parallel(model)
		for disc in discriminators:
			discriminators[disc][0] = torch.nn.DataParallel(discriminators[disc][0])

	model.to(args.device)
	for disc in discriminators:
			discriminators[disc][0].to(args.device)

	print("model built", file=sys.stderr)
	#print("model created")
	train_loader, val_loader, test_loader = train_util.get_dataloaders(args)
	print("loaders acquired", file=sys.stderr)
	#print("loaders acquired")

	model_name = f"vae_{args.recons_loss}"
	if args.output_folder is None:
		args.output_folder = os.path.join(model_name, args.dataset, f"depth_{args.enc_type}_{args.dec_type}_hs_{args.img_res}_{args.hidden_size}")

	log_save_path = os.path.join("./logs", args.output_folder)
	model_save_path = os.path.join("./models", args.output_folder)

	if not os.path.exists(log_save_path):
		os.makedirs(log_save_path)
		print(f"log:{log_save_path}", file=sys.stderr)
		sys.stderr.flush()
	if not os.path.exists(model_save_path):
		os.makedirs(model_save_path)


	writer = SummaryWriter(log_save_path)

	print(f"train loader length:{len(train_loader)}", file=sys.stderr)
	best_loss = torch.tensor(np.inf)
	
	if args.weights == "load":
		start_epoch = train_util.load_state(model_save_path, model, opt, discriminators)
	else:
		start_epoch = 0

	recons_input_img = train_util.log_input_img_grid(test_loader, writer)

	train_util.log_recons_img_grid(recons_input_img, model, 0, args.device, writer)

	stop_patience = args.stop_patience
	for epoch in range(start_epoch, args.num_epochs):
		
		try:
			train(model, train_loader, opt, epoch, writer, args, discriminators)
		except RuntimeError as err:
			print("".join(traceback.TracebackException.from_exception(err).format()), file=sys.stderr)
			print("*******", file=sys.stderr)
			print(err, file=sys.stderr)
			exit(0)


		val_loss_dict, z = train_util.test(get_losses, model, val_loader, args, discriminators)
		print(f"epoch loss:{val_loss_dict['recons_loss'].item()}")


		train_util.save_recons_img_grid("test", recons_input_img, model, epoch+1, args)
		train_util.save_interp_img_grid("test", recons_input_img, model, epoch+1, args)

		train_util.log_losses("val", val_loss_dict, epoch+1, writer)
		train_util.log_latent_metrics("val", z, epoch+1, writer)
		train_util.save_state(model, opt, discriminators, val_loss_dict["recons_loss"], best_loss, args.recons_loss, epoch, model_save_path)

	# 	#early stop check
	# 	# if val_loss_dict["recons_loss"] - best_loss < args.threshold:
	# 	# 	stop_patience -= 1
	# 	# else:
	# 	# 	stop_patience = args.stop_patience
		
	# 	# if stop_patience == 0:
	# 	# 	print("training early stopped!")
	# 	# 	break

	# 	ae_lr_scheduler.step(val_loss_dict["recons_loss"])
	# 	if args.recons_loss != "mse":
	# 		recons_disc_lr_scheduler.step(val_loss_dict["recons_disc_loss"])

	# end_time = datetime.today()

	# print(f"run duration:{end_time-start_time}", file=sys.stderr)


def val_test(args):
	writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
	save_filename = './models/{0}'.format(args.output_folder)

	train_loader, valid_loader, test_loader = train_util.get_dataloaders(args)
	recons_input_img = train_util.log_input_img_grid(test_loader, writer)

	input_dim = 3
	model = VAE(input_dim, args.hidden_size, args.enc_type, args.dec_type)
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
	
		# print(z.shape)
		train_util.log_recons_img_grid(recons_input_img, model, epoch+1, args.device, writer)
		train_util.log_interp_img_grid(recons_input_img, model, epoch+1, args.device, writer)

		train_util.log_losses("val", val_loss_dict, epoch+1, writer)
		train_util.log_latent_metrics("val", z, epoch+1, writer)
		train_util.save_state(model, optimizer, discriminators, val_loss_dict["recons_loss"], best_loss, args.recons_loss, epoch, save_filename)


	print(val_loss_dict)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# optim
	parser.add_argument("--num_epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument('--lr', type=float, default=2e-4)
	parser.add_argument("--disc_lr", type=float, default=None)
	parser.add_argument("--threshold", type=float, default=1.0,
		help="val loss reduction threshold")
	parser.add_argument("--lr_patience", type=int, default=2,
		help="epochs of patience for val loss reduction under threshold before reducing learning rate")
	parser.add_argument("--stop_patience", type=int, default=3,
		help="epochs of patience for val loss reduction under threshold before early stop")
	
	# data
	parser.add_argument("--dataset", type=str, default="imagenet", 
		choices={'imagenet', "CIFAR10", "MNIST", "FashionMNIST", "tiny-imagenet"})
	parser.add_argument("--data_folder", type=str,
		help="location of dataset")
	parser.add_argument("--img_res", type=int, default=128,
		help='image resolution to random crop')
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
	parser.add_argument("--hidden_size", type=int, default=256)


	# Miscellaneous
	parser.add_argument("--output_folder", type=str)
	parser.add_argument('--num_workers', type=int, default=0,
		help='number of workers for trajectories sampling (default: {0})'.format(mp.cpu_count() - 1))
	parser.add_argument('--device', type=str, default='cuda', choices={"cpu", "cuda", "cuda:0", "cuda:1"})
	# parser.add_argument("--num_gpu", type=int, default=torch.cuda.device_count, help="number of gpus to train with")
	args = parser.parse_args()

	if args.disc_lr is None:
		args.disc_lr = args.lr

	#print("chosen", args.device)
	args.device = torch.device(args.device
		if torch.cuda.is_available() else 'cpu')
	print("set", args.device, file=sys.stderr)

	print("training vae with following params", file=sys.stderr)
	print(f"batch size: {args.batch_size}", file=sys.stderr)
	print(f"encoder:{args.enc_type}", file=sys.stderr)
	print(f"decoder:{args.dec_type}", file=sys.stderr)
	print(f"loss:{args.recons_loss}", file=sys.stderr)
	print(f"mode:{args.weights}")
	print("testing print", file=sys.stderr)
	args.steps = 0
	main(args)
	#val_test(args)
	#generate_samples()
