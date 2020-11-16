"""
	code modified from https://gist.github.com/kylemcdonald/e8ca989584b3b0e6526c0a737ed412f0
"""
import pdb
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid

from modules import ACAI, to_scalar

from tensorboardX import SummaryWriter

from PIL import Image
import time

def swap_halves(batch):
	if batch.shape[0]%2 == 0:
		a, b = batch.split(batch.shape[0]//2)
		return torch.cat([b, a])
	else:
		a, b, c = batch.split(batch.shape[0]//2)
		return torch.cat([c, b, a])

def train(epoch, data_loader, model, optimizer, args, writer):
	model.train()
	for idx, (images, _) in enumerate(data_loader):
		images = images.to(args.device)

		optimizer.zero_grad()
		x_tilde, z = model(images)
		
		############1111 AE loss 11111############
		alpha = torch.rand(images.shape[0], 1, 1, 1).to(args.device)
		z_interp = torch.lerp(z, swap_halves(z), alpha)
		x_tilde_interp = model.decoder(z_interp)

		assert x_tilde_interp.shape[2] == images.shape[2] and x_tilde_interp.shape[3] == images.shape[3]

		# Reconstruction loss
		loss_recons = F.mse_loss(x_tilde, images)

		# perceptual/discriminator loss
		#pdb.set_trace()
		loss_perc = model.disc_forward(x_tilde_interp)
		loss_perc = torch.mean(loss_perc**2)

		loss_ae = loss_recons + args.lam * loss_perc

		loss_ae.backward(retain_graph=True)
		###########11111 ----------- 111111#########

		###########22222 critic loss 222222#########

		pred_alpha = model.disc_forward(x_tilde_interp)

		# interpolation of recons and orig in data space
		x_interp = torch.lerp(x_tilde, images, args.gamma).to(args.device)

		loss_alpha = F.mse_loss(pred_alpha, alpha.squeeze())
		loss_reg = model.disc_forward(x_interp)
		loss_reg = torch.mean(loss_reg**2)

		loss_disc = loss_alpha + loss_reg
		loss_disc.backward()
		##########22222 ----------- 222222##########


		# Logs
		if idx % 1000 == 0:
			print("iter:", str(epoch*(len(data_loader))+idx)+"/"+str(args.num_epochs*(len(data_loader))), "iter loss:", loss_ae.item()+loss_disc.item())
		writer.add_scalar('loss/train/reconstruction', loss_ae.item(), args.steps)
		writer.add_scalar('loss/train/discriminator', loss_disc.item(), args.steps)

		optimizer.step()
		args.steps += 1

def test(data_loader, model, args, writer):
	model.eval()
	with torch.no_grad():
		loss_recons, loss_vq = 0., 0.
		for images, _ in data_loader:

			images = images.to(args.device)
			x_tilde, z = model(images)

			# interpolation of recons and orig in data space
			x_interp = torch.lerp(x_tilde, images, args.gamma).to(args.device)

			### AE loss ###
			alpha = torch.rand(images.shape[0], 1, 1, 1).to(args.device)
			z_interp = torch.lerp(z, swap_halves(z), alpha)
			x_tilde_interp = model.decoder(z_interp)

			# Reconstruction loss
			loss_recons = F.mse_loss(x_tilde, images)

			# perceptual/discriminator loss
			loss_perc = model.disc(x_tilde_interp)
			loss_perc = torch.mean(loss_perc**2)

			loss_ae = loss_recons + loss_perc

			### critic loss ###

			pred_alpha = model.disc(x_tilde_interp)

			loss_alpha = F.mse_loss(pred_alpha, alpha.flatten())
			loss_reg = model.disc_forward(x_interp)
			loss_reg = torch.mean(loss_reg**2)

			loss_disc = loss_alpha + loss_reg

		loss_recons /= len(data_loader)
		loss_vq /= len(data_loader)

	return loss_ae.item(), loss_disc.item()

def generate_samples(images, model, args):
	with torch.no_grad():
		images = images.to(args.device)
		x_tilde, _ = model(images)
	return x_tilde

def main(args):
	writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
	save_filename = './models/{0}'.format(args.output_folder)
	pixel_mean=[0.485, 0.456, 0.406]
	pixel_std=[0.229, 0.224, 0.225]
	if args.dataset in ['mnist', 'fashion-mnist', 'cifar10']:
		transform = transforms.Compose([
			transforms.ToTensor(),
			#transforms.Normalize(mean=pixel_mean, std=pixel_std)
			#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		if args.dataset == 'cifar10':
			# Define the train & test datasets
			train_dataset = datasets.CIFAR10(args.data_folder,
				train=True, download=True, transform=transform)
			test_dataset = datasets.CIFAR10(args.data_folder,
				train=False, transform=transform)
			num_channels = 3
		valid_dataset = test_dataset
	elif args.dataset == 'tiny-imagenet':
		im_transform = transforms.Compose([transforms.ToTensor()])#, 
					   #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		train_dataset = datasets.ImageFolder(root=os.path.join(args.data_folder, "train"), transform=im_transform )
		test_dataset = datasets.ImageFolder(root=os.path.join(args.data_folder, "test"), transform=im_transform )
		valid_dataset = datasets.ImageFolder(root=os.path.join(args.data_folder, "val"), transform=im_transform )
		num_channels = 3
	elif args.dataset == 'imagenet':
		print("img res", args.img_res)
		im_transform = transforms.Compose([ \
						transforms.CenterCrop(args.img_res), \
						transforms.ToTensor()])
		train_dataset = datasets.ImageFolder(root=os.path.join(args.data_folder, "train"), transform=im_transform )
		test_dataset = datasets.ImageFolder(root=os.path.join(args.data_folder, "val"), transform=im_transform )
		valid_dataset = test_dataset
		num_channels = 3

	# Define the data loaders
	train_loader = torch.utils.data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True,
		num_workers=args.num_workers, pin_memory=True)
	valid_loader = torch.utils.data.DataLoader(valid_dataset,
		batch_size=args.batch_size, shuffle=False, drop_last=True,
		num_workers=args.num_workers, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_dataset,
		batch_size=16, shuffle=False)

	# Fixed images for Tensorboard
	#print(next(iter(test_loader)))
	#print(len(iter(train_loader)))
	fixed_images, _ = next(iter(test_loader))
	fixed_grid = make_grid(fixed_images, nrow=8)#norm*, range=(-1, 1), normalize=True)
	writer.add_image('original', fixed_grid, 0)

	model = ACAI(args.img_res, num_channels, args.hidden_size).to(args.device)
	
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	# Generate the samples first once
	reconstruction = generate_samples(fixed_images, model, args)
	grid = make_grid(reconstruction.cpu(), nrow=8)#norm*, range=(-1, 1), normalize=True)
	writer.add_image('reconstruction', grid, 0)


	best_loss = -1.
	for epoch in range(args.num_epochs):
		train(epoch, train_loader, model, optimizer, args, writer)
		loss, loss_disc = test(valid_loader, model, args, writer)
		writer.add_scalar('loss/test/ae', loss, epoch+1)
		writer.add_scalar('loss/test/disc', loss_disc, epoch+1)


		reconstruction = generate_samples(fixed_images, model, args)
		grid = make_grid(reconstruction.cpu(), nrow=8)#norm*, range=(-1, 1), normalize=True)
		writer.add_image('reconstruction', grid, epoch + 1)

		print("trained epoch:", epoch)
		if (epoch == 0) or (loss < best_loss):
			best_loss = loss
			print("new best loss:", loss)
			with open('{0}/best.pt'.format(save_filename), 'wb') as f:
				torch.save(model.state_dict(), f)
		with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
			torch.save(model.state_dict(), f)

if __name__ == '__main__':
	import argparse
	import os
	import multiprocessing as mp

	start_time = time.time()

	parser = argparse.ArgumentParser(description='ACAI')

	# General
	parser.add_argument('--data-folder',default="/shared/kgcoe-research/mil/ImageNet/", type=str,
		help='name of the data folder')
	parser.add_argument('--dataset', default="imagenet", type=str, choices={"cifar10", "tiny-imagenet", "imagenet"},
		help='name of the dataset ')
	parser.add_argument('--img_res', default=128, type=int,
		help='image resolution to center crop')

	# model
	parser.add_argument('--enc_type', type=str, choices={"shallow", "moderate_shallow", "moderate", "deep"},
		help='depth of encoder')
	parser.add_argument('--dec_type', type=int, choices={"shallow", "moderate_shallow", "moderate", "deep"},
		help='depth of decoder')
	parser.add_argument('--hidden-size', type=int, default=256,
		help='size of the latent vectors (default: 256)')

	# Optimization
	parser.add_argument('--batch-size', type=int, default=60,
		help='batch size (default: 128)')
	parser.add_argument('--num-epochs', type=int, default=30,
		help='number of epochs (default: 100)')
	parser.add_argument('--lr', type=float, default=2e-4,
		help='learning rate for Adam optimizer (default: 2e-4)')
	parser.add_argument('--gamma', '-g', type=float, default=0.2,
		help='regularization for critic to perform well with bad reconstructions')
	parser.add_argument('--lam', '-l', type=float, default=0.5,
		help='weight for loss from critic')

	# Miscellaneous
	parser.add_argument('--output-folder', type=str,
		help='name of the output folder')
	parser.add_argument('--num-workers', type=int, default=0,
		help='number of workers for trajectories sampling (default: 0)')
	parser.add_argument('--device', type=str, default='cuda',
		help='set the device (cpu or cuda, default: cpu)')

	args = parser.parse_args()

	# Create logs and models folder if they don't exist
	if not os.path.exists('./logs'):
		os.makedirs('./logs')
	if not os.path.exists('./models'):
		os.makedirs('./models')
	# Device
	print("chosen", args.device)
	args.device = torch.device(args.device
		if torch.cuda.is_available() else 'cpu')
	print("set", args.device)
	# Slurm

	print(args.output_folder)
	if args.output_folder is None:
		args.output_folder = f"acai/{args.dataset}/depth_{args.enc_type}_{args.dec_type}_hs_{args.img_res}_{args.hidden_size}"

	if 'SLURM_JOB_ID' in os.environ:
		args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
	if not os.path.exists('./models/{0}'.format(args.output_folder)):
		os.makedirs('./models/{0}'.format(args.output_folder))
	args.steps = 0

	main(args)

	with open("time.txt", "w") as f:
		f.write(str(time.time() - start_time))