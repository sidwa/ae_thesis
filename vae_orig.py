
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

from modules import VAE, Discriminator, ClubbedPermutationComparator, FullPermutationComparator, AnchorComparator
import train_util

BATCH_SIZE = 32
N_EPOCHS = 30
PRINT_INTERVAL = 500
# DATASET = 'FashionMNIST'  # CIFAR10 | MNIST | FashionMNIST
# NUM_WORKERS = 4

INPUT_DIM = 3
DIM = 128
# Z_DIM = 128
LR = 1e-3


def train(model, opt, train_loader):
	train_loss = []
	model.train()
	for batch_idx, data in enumerate(train_loader):
		start_time = time.time()
		x = data[0].cuda()

		x_tilde, _, kl_d = model(x)

		# pdb.set_trace()
		loss_recons = F.mse_loss(x_tilde, x, reduction="sum") / x.shape[0]

		loss = loss_recons + kl_d

		nll = -Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x)
		log_px = nll.mean().item() - np.log(128) + kl_d.item()
		log_px /= np.log(2)

		opt.zero_grad()
		loss.backward()
		opt.step()

		train_loss.append([log_px, loss.item()])

		if (batch_idx + 1) % PRINT_INTERVAL == 0:
			print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {:5.3f} ms/batch'.format(
				batch_idx * len(x), len(train_loader.dataset),
				PRINT_INTERVAL * batch_idx / len(train_loader),
				np.asarray(train_loss)[-PRINT_INTERVAL:].mean(0),
				1000 * (time.time() - start_time)
			))


def val(model, val_loader):
	start_time = time.time()
	val_loss = []
	model.eval()
	with torch.no_grad():
		for batch_idx, data in enumerate(val_loader):
			x = data[0].cuda()
			x_tilde, _, kl_d = model(x)
			loss_recons = F.mse_loss(x_tilde, x, size_average=False) / x.size(0)
			loss = loss_recons + kl_d
			val_loss.append(loss.item())

	print('\nValidation Completed!\tLoss: {:5.4f} Time: {:5.3f} s'.format(
		np.asarray(val_loss).mean(0),
		time.time() - start_time
	))
	return np.asarray(val_loss).mean(0)


def generate_reconstructions(model, epoch, split, test_loader):
	model.eval()
	data = test_loader.__iter__().next()
	x = data[0]
	x = x[:16].cuda()
	x_tilde, _ = model(x)

	x_cat = torch.cat([x, x_tilde], 0)
	images = x_cat.cpu().data

	save_image(
		images,
		f'samples/{split}/vae_reconstructions_tiny-imagenet_{epoch}.png',
		nrow=8
	)




def main(args):

	train_loader, val_loader, test_loader = train_util.get_dataloaders(args)
	input_dim = 3
	model = VAE(input_dim, args.hidden_size, args.enc_type, args.dec_type)
	opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)


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

	train_util.save_recons_img_grid("val", recons_input_img, model, 0, args)


	for epoch in range(1, args.num_epochs):
		print("Epoch {}:".format(epoch))
		train(model, opt, train_loader)
		curr_loss = val(model, val_loader)
		# val_loss_dict, z = train_util.test(get_losses, model, val_loader, args, discriminators)

		print(f"epoch val loss:{curr_loss}", file=sys.stderr)
		sys.stderr.flush()
		train_util.save_recons_img_grid("val", recons_input_img, model, epoch+1, args)
		train_util.save_interp_img_grid("val", recons_input_img, model, epoch+1, args)

		# generate_reconstructions(model, epoch, "test", test_loader)
		# generate_reconstructions(model, epoch, "train", train_loader)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# optim
	parser.add_argument("--num_epochs", type=int, default=20)
	parser.add_argument("--batch_size", type=int, default=64)
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
	print(f"batch size: {BATCH_SIZE}", file=sys.stderr)
	print(f"encoder:{args.enc_type}", file=sys.stderr)
	print(f"decoder:{args.dec_type}", file=sys.stderr)
	print(f"loss:{args.recons_loss}", file=sys.stderr)
	print(f"mode:{args.weights}")
	args.steps = 0
	main(args)
	#val_test(args)
	#generate_samples()

