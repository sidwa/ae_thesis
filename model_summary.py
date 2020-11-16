"""
	code file to print model and the memory requirement.
	Also uses the memory requirement to output maximum batch_size that would completely fill 
	available vram of gpu
"""

import pickle
import re
import sys
import pdb

import torchsummary
from modules import ACAI, VectorQuantizedVAE, VAE, Discriminator, \
					AnchorComparator, ClubbedPermutationComparator, FullPermutationComparator

import torch


def model_summary(model_type, img_res, hidden_size, enc_type, dec_type, loss, batch_size, device=torch.device("cuda:1"), verbose=True):
	pattern = re.compile(r"Params size \(MB\):(.*)\n")
	pattern2 = re.compile(r"Forward/backward pass size \(MB\):(.*)\n")
	input_dim = 3
	enc_input_size = (input_dim, img_res, img_res)
	dec_input_size = (hidden_size, img_res//4, img_res//4)
	pdb.set_trace()
	if verbose:
		print(f"model:{model_type}")
		print(f"depth:{enc_type}_{dec_type}")

	if model_type == "acai":
		model = ACAI(img_res, input_dim, hidden_size, enc_type, dec_type).to(device)
	elif model_type == "vqvae":
		model = VectorQuantizedVAE(input_dim, hidden_size, enc_type=enc_type, dec_type=dec_type).to(device)
	elif model_type == "vae":
		model = VAE(input_dim, hidden_size, enc_type=enc_type, dec_type=dec_type).to(device)

	encoder_summary, _= torchsummary.summary_string(model.encoder, enc_input_size, device=device, batch_size=batch_size)
	decoder_summary, _= torchsummary.summary_string(model.decoder, dec_input_size, device=device, batch_size=batch_size)
	if verbose:
		print(encoder_summary)
		print(decoder_summary)

	discriminators = {}

	if model_type == "acai":
		disc = Discriminator(input_dim, img_res, "image").to(device)
	
		disc_summary, _= torchsummary.summary_string(disc, enc_input_size, device=device, batch_size=batch_size )
		disc_param_size = float(re.search(pattern, disc_summary).group(1))
		disc_forward_size = float(re.search(pattern2, disc_summary).group(1))
		discriminators["interp_disc"] = (disc_param_size, disc_forward_size)
	if loss == "gan":
		disc = Discriminator(input_dim, img_res, "image").to(device)
	
		disc_summary, _= torchsummary.summary_string(disc, enc_input_size, device=device, batch_size=batch_size )
		disc_param_size = float(re.search(pattern, disc_summary).group(1))
		disc_forward_size = float(re.search(pattern2, disc_summary).group(1))
		discriminators["recons_disc"] = (disc_param_size, 2*disc_forward_size)
	elif loss == "comp":
		disc = AnchorComparator(input_dim*2, img_res, "image").to(device)
	
		disc_summary, _= torchsummary.summary_string(disc, enc_input_size, device=device, batch_size=batch_size )
		disc_param_size = float(re.search(pattern, disc_summary).group(1))
		disc_forward_size = float(re.search(pattern2, disc_summary).group(1))
		discriminators["recons_disc"] = (disc_param_size, 2*disc_forward_size)
	elif "comp_2" in loss:
		disc = ClubbedPermutationComparator(input_dim*2, img_res, "image").to(device)
	
		disc_summary, _= torchsummary.summary_string(disc, enc_input_size, device=device, batch_size=batch_size )
		disc_param_size = float(re.search(pattern, disc_summary).group(1))
		disc_forward_size = float(re.search(pattern2, disc_summary).group(1))
		discriminators["recons_disc"] = (disc_param_size, 2*disc_forward_size)
	elif"comp_6" in loss:
		disc = FullPermutationComparator(input_dim*2, img_res, "image").to(device)
	
		disc_summary, _= torchsummary.summary_string(disc, enc_input_size, device=device, batch_size=batch_size )
		disc_param_size = float(re.search(pattern, disc_summary).group(1))
		disc_forward_size = float(re.search(pattern2, disc_summary).group(1))
		discriminators["recons_disc"] = (disc_param_size, 2*disc_forward_size)

	encoder_param_size = float(re.search(pattern, encoder_summary).group(1))
	encoder_forward_size = float(re.search(pattern2, encoder_summary).group(1))
	decoder_param_size = float(re.search(pattern, decoder_summary).group(1))
	decoder_forward_size = float(re.search(pattern2, decoder_summary).group(1))

	if verbose:
		if "ACAI" in str(type(model)):
			print(f"discriminator:\n\tparams:{disc_param_size}\n\tforward:{disc_forward_size}")
		
		if loss == "gan":
			print(f"reconstruction discriminator:\n\tparams:{disc_param_size}\n\tforward:{disc_forward_size}")

		print(f"encoder:\n\tparams:{encoder_param_size}\n\tforward:{encoder_forward_size}")
		print(f"decoder:\n\tparams:{decoder_param_size}\n\tforward:{decoder_forward_size}")

	encoder = {"params":encoder_param_size, "forward":encoder_forward_size}
	decoder = {"params":decoder_param_size, "forward":decoder_forward_size}

	return encoder, decoder, discriminators

def get_batch_size(model, img_res, hidden_size, enc_type, dec_type, loss, vram=8*1024, device=torch.device("cuda:1")):
	
	encoder, decoder, discriminators = model_summary(model, img_res, hidden_size, enc_type, dec_type, loss, -1, device=device, verbose=False)

	fixed_size = encoder["params"] + decoder["params"]

	for disc_type in discriminators:
		fixed_size += discriminators[disc_type][0]

	forward_size = encoder["forward"] + decoder["forward"]

	for disc_type in discriminators:
		forward_size += discriminators[disc_type][1]

	batch_ram = vram - fixed_size

	batch_size = batch_ram // forward_size

	return batch_size
	


def test():
	ACAI = ACAI(3, 256, "shallow", "shallow").to("cuda:1")
	model_summary(vae, 128, 256, "shallow", "shallow", "mse", 47)
	# print(get_batch_size("vae", 128, 256, "shallow", "shallow", "comp_2_dc"))


# def create_model_summary_dict():
# 	"""
# 		creates a dictionary of with key containing model type with depth and latent size
# 		value as the optimum batch size.
# 	"""
# 	dim = [256, 128, 64, 48, 32]
# 	depth = ["shallow", "moderate_shallow", "moderate", "deep"]
# 	input_dim = 3
# 	img_res = 128
# 	device = torch.device("cuda:1")
# 	vram = 32 * 1024
# 	losses = ["mse", "gan"]
# 	batch_size = {}
# 	idx = 0
# 	for enc_type in depth:
# 		for dec_type in depth:
# 			for hidden_size in dim:
# 				for loss in losses:
# 					print(f"{idx}/{len(depth)*len(depth)*len(dim)*len(losses)}", file=sys.stderr)
					
# 					acai = ACAI(img_res, input_dim, hidden_size, enc_type, dec_type).to(device)
# 					key = f"acai_{enc_type}_{dec_type}_{hidden_size}_{loss}"
# 					batch_size[key] = get_batch_size(acai, img_res, hidden_size, enc_type, dec_type, loss, vram=vram, device=device)

# 					vae = VAE(input_dim, hidden_size, enc_type, dec_type).to(device)
# 					key = f"vae_{enc_type}_{dec_type}_{hidden_size}_{loss}"
# 					batch_size[key] = get_batch_size(vae, img_res, hidden_size, enc_type, dec_type, loss, vram=vram, device=device)

# 					vqvae = VectorQuantizedVAE(input_dim, hidden_size, enc_type=enc_type, dec_type=dec_type).to(device)
# 					key = f"vqvae_{enc_type}_{dec_type}_{hidden_size}_{loss}"
# 					batch_size[key] = get_batch_size(vqvae, img_res, hidden_size, enc_type, dec_type, loss, vram=vram, device=device)
					
# 					idx += 1

# 	for key in batch_size:
# 		print(f"model:{key}\tbatch size:{batch_size[key]}")

# 	with open("model_batch_size_v100", "wb") as bs:
# 		pickle.dump(batch_size, bs)

def main():
	print("main!!!")
	device = torch.device("cuda")
	# acai = ACAI(128, 3, 256, "shallow", "shallow").to(torch.device(device))
	model_summary("acai", 128, 256, "shallow", "shallow", "gan", vram=32*1024, device=device)
	print(get_batch_size("acai", 128, 256, "shallow", "shallow", "gan", vram=32*1024, device=device))
	# create_model_summary_dict()

if __name__ == "__main__":
	main()
