import sys
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from functions import vq, vq_st

def to_scalar(arr):
	if type(arr) == list:
		return [x.item() for x in arr]
	else:
		return arr.item()



class LatentMultiClassDiscriminator(nn.Module):
	"""
		Use this class for perturbation features with more than 2 classes.
		eg: rotation if there are 4 possible angles (0, 90, 180, 270)

		:param hidden_size: depth of latent representation volume 
		:param latent_dim: height and width of latent representation volume
		:param num_perturb_class: number of ways a specific pertrubation can be performed.
									(4 for the example given above)
	"""
	def __init__(self, hidden_size, latent_dim, num_perturb_class, input_type="image"):
		super(LatentMultiClassDiscriminator, self).__init__()

		self.conv = AutoEncoder._get_conv_type(input_type)

		self.net = nn.Sequential(
			self.conv(hidden_size, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			self.conv(128, 256, 4, 2, 1),
			ResBlock(256, input_type),
			self.conv(256, 512, 3, 1, 1),
			ResBlock(512, input_type),
			self.conv(512, 1024, 3, 1, 1),
			ResBlock(1024, input_type),
			self.conv(1024, 512, 3, 1, 1),
			nn.Flatten(),
			nn.Linear((latent_dim//4)*(latent_dim//4)*512, 256),
			nn.Linear(256, 100),
			nn.Linear(100, num_perturb_class),
			nn.Softmax(dim=1)
		)
	

	def forward(self, inp):

		# for layer in self.net:
		# 	pdb.set_trace()
		# 	inp = layer(inp)
		# 	print(layer)
		# 	print(inp.shape)

		# return inp

		return self.net(inp)

class Latent2ClassDiscriminator(nn.Module):
	def __init__(self, hidden_size, latent_dim, input_type="image"):
		super(Latent2ClassDiscriminator, self).__init__()

		self.conv = AutoEncoder._get_conv_type(input_type)

		self.net = nn.Sequential(
			self.conv(hidden_size, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			self.conv(128, 256, 4, 2, 1),
			ResBlock(256, input_type),
			self.conv(256, 512, 3, 1, 1),
			ResBlock(512, input_type),
			self.conv(512, 1024, 3, 1, 1),
			ResBlock(1024, input_type),
			self.conv(1024, 512, 3, 1, 1),
			nn.Flatten(),
			nn.Linear((latent_dim//4)*(latent_dim//4)*512, 256),
			nn.Linear(256, 100),
			nn.Linear(100, 1),
		)

	

	def forward(self, inp):



		return self.net(inp)



class Discriminator(nn.Module):
	def __init__(self, input_dim, img_res, input_type="image"):
		super(Discriminator, self).__init__()

		self.conv = AutoEncoder._get_conv_type(input_type)

		self.net = nn.Sequential(
			self.conv(input_dim, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			self.conv(128, 256, 4, 2, 1),
			ResBlock(256, input_type),
			self.conv(256, 512, 4, 2, 1),
			ResBlock(512, input_type),
			self.conv(512, 256, 4, 2, 1),
			ResBlock(256, input_type),
			self.conv(256, 256, 4, 2, 1),
			nn.Flatten(),
			nn.Linear((img_res//32)*(img_res//32)*256, 256),
			nn.Linear(256, 100),
			nn.Linear(100, 1),
		)

	def forward(self, inp):
		return self.net(inp)

class AnchorComparator(nn.Module):
	def __init__(self, input_dim, img_res, input_type="image"):
		super(AnchorComparator, self).__init__()

		self.conv = AutoEncoder._get_conv_type(input_type)

		self.net = nn.Sequential(
			self.conv(input_dim, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			self.conv(128, 256, 4, 2, 1),
			ResBlock(256, input_type),
			self.conv(256, 512, 4, 2, 1),
			ResBlock(512, input_type),
			self.conv(512, 1024, 4, 2, 1),
			ResBlock(1024, input_type),
			self.conv(1024, 512, 4, 2, 1),
			nn.Flatten(),
			nn.Linear((img_res//32)*(img_res//32)*512, 256),
			nn.Linear(256, 100),
			nn.Linear(100, 1),
		)

	def forward(self, inp):
		return self.net(inp)


class ClubbedPermutationComparator(nn.Module):
	def __init__(self, input_dim, img_res, input_type="image"):
		super(ClubbedPermutationComparator, self).__init__()

		self.conv = AutoEncoder._get_conv_type(input_type)

		self.net = nn.Sequential(
			self.conv(input_dim, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			self.conv(128, 256, 4, 2, 1),
			ResBlock(256, input_type),
			self.conv(256, 512, 4, 2, 1),
			ResBlock(512, input_type),
			self.conv(512, 1024, 4, 2, 1),
			ResBlock(1024, input_type),
			self.conv(1024, 512, 4, 2, 1),
			nn.Flatten(),
			nn.Linear((img_res//32)*(img_res//32)*512, 256),
			nn.Linear(256, 100),
			nn.Linear(100, 2),
		)
	
	def forward(self, inp):
		return self.net(inp)

class FullPermutationComparator(nn.Module):
	def __init__(self, input_dim, img_res, input_type="image"):
		super(FullPermutationComparator, self).__init__()

		self.conv = AutoEncoder._get_conv_type(input_type)

		self.net = nn.Sequential(
			self.conv(input_dim, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			self.conv(128, 256, 4, 2, 1),
			ResBlock(256, input_type),
			self.conv(256, 512, 4, 2, 1),
			ResBlock(512, input_type),
			self.conv(512, 1024, 4, 2, 1),
			ResBlock(1024, input_type),
			self.conv(1024, 512, 4, 2, 1),
			nn.Flatten(),
			nn.Linear((img_res//32)*(img_res//32)*512, 256),
			nn.Linear(256, 100),
			nn.Linear(100, 6),
		)
		
	def forward(self, inp):
		return self.net(inp)


class ChannelPredLayer(nn.Module):
	def __init__(self, embed_size):
		super(ChannelPredLayer, self).__init__()
		
		self.channel_color_layer = nn.Linear(embed_size, 6*3)
		self.channel_type_layer = nn.Linear(embed_size, 6)

	
	def forward(self, emb):
		color_pred = self.channel_color_layer(emb)
		type_pred = self.channel_type_layer(emb)

		color_pred = color_pred.reshape(-1, 3, 6)
		color_pred = nn.functional.softmax(color_pred, dim=1)
		return color_pred, type_pred
	


class FullPermutationColorComparator(nn.Module):
	def __init__(self, input_dim, img_res, input_type="image"):
		super(FullPermutationColorComparator, self).__init__()

		self.conv = AutoEncoder._get_conv_type(input_type)

		self.net = nn.Sequential(
			self.conv(input_dim, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			self.conv(128, 256, 4, 2, 1),
			ResBlock(256, input_type),
			self.conv(256, 512, 4, 2, 1),
			ResBlock(512, input_type),
			self.conv(512, 1024, 4, 2, 1),
			ResBlock(1024, input_type),
			self.conv(1024, 512, 4, 2, 1),
			nn.Flatten(),
			nn.Linear((img_res//32)*(img_res//32)*512, 256),
			nn.Linear(256, 100),
			ChannelPredLayer(100)
		)


		
	def forward(self, inp):
		color_pred, chtype_pred = self.net(inp)
		return torch.squeeze(color_pred), torch.squeeze(chtype_pred)


class AutoEncoder(nn.Module):

	def __init__(self):
		super(AutoEncoder, self).__init__()

	def weights_init(self, m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			try:
				nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
				m.bias.data.fill_(0)
			except AttributeError:
				print("Skipping initialization of ", classname)

	@staticmethod
	def _get_conv_type(input_type):
		"""
			returns conv type based on type of data
		"""
		if input_type == "image":
			conv = nn.Conv2d
		elif input_type == "audio":
			conv = nn.Conv1d
		else:
			print("invalid input type!")
			conv = None
		
		return conv

	@staticmethod
	def _get_deconv_type(input_type):
		"""
			returns conv type based on type of data
		"""
		if input_type == "image":
			conv = nn.ConvTranspose2d
		elif input_type == "audio":
			conv = nn.ConvTranspose1d
		else:
			print("invalid input type!")
			conv = None
		
		return conv

	@staticmethod
	def get_encoder(input_dim, dim, enc_type="shallow", num_res_blocks=None, res_block_sep=0,input_type="image"):
		"""
			single function for getting encoder, maintain common baseline for all
			AEs.

			input_dim: number of channels in input
			dim: dim of latent vector
			enc_type: specify how deep the encoder would be shallow|moderate_shallow|moderate|deep
			num_res_blocks: number of residual blocks in encoder, 
							granular control for depth, allows 
							control of depth based on number of resblocks
			res_block_sep: number of resblocks between downsample layers( stride 2 conv layers )
			input_type: data modality that encoder accepts image|audio
			

		"""
		#pdb.set_trace()
		if num_res_blocks is None:
			if enc_type == "shallow":
				num_res_blocks = 2
			elif enc_type == "moderate_shallow":
				num_res_blocks = 10
			elif enc_type == "moderate":
				num_res_blocks = 20
			elif enc_type == "deep":
				num_res_blocks = 30
			
			res_block_sep = num_res_blocks // 2
		
		if res_block_sep < 0 or res_block_sep > num_res_blocks:
			print("res block separation cannot be less than totoal res blocks", file=sys.stderr)

		conv = AutoEncoder._get_conv_type(input_type)
		
		downsample_layers_1 = [conv(input_dim, dim // 2, 4, 2, 1),
							nn.BatchNorm2d(dim // 2),
							nn.ReLU(True)]

		downsample_layers_2 = [conv(dim // 2, dim, 4, 2, 1),
							nn.BatchNorm2d(dim),
							nn.ReLU(True)]

		# downsample_layers_3 = [conv(dim // 2, dim, 4, 2, 1),
		# 					nn.BatchNorm2d(dim),
		# 					nn.ReLU(True)]
		layers = []

		layers += downsample_layers_1

		while res_block_sep > 0:
			layers.append(ResBlock(dim // 2, input_type))
			res_block_sep -= 1
			num_res_blocks -= 1
		
		layers += downsample_layers_2

		while num_res_blocks > 0:
			layers.append(ResBlock(dim, input_type))
			num_res_blocks -= 1
		
		return nn.Sequential(*layers)


	@staticmethod
	def get_decoder(input_dim, dim, dec_type="shallow", num_res_blocks=None, res_block_sep=0, input_type="image"):
		
		"""
			single function for getting decoder, maintain common baseline for all
			AEs.

			input_dim: number of channels in input
			dim: dim of latent vector
			dec_type: specify how deep the encoder would be shallow|moderate_shallow|moderate|deep
			num_res_blocks: number of residual blocks in encoder, 
							granular control for depth, allows 
							control of depth based on number of resblocks
			res_block_sep: number of resblocks between downsample layers( stride 2 conv layers )
			input_type: data modality that encoder accepts image|audio
			

		"""

		if num_res_blocks is None:
			if dec_type == "shallow":
				num_res_blocks = 2
			elif dec_type == "moderate_shallow":
				num_res_blocks = 10
			elif dec_type == "moderate":
				num_res_blocks = 20
			elif dec_type == "deep":
				num_res_blocks = 30
			
			res_block_sep = num_res_blocks // 2
		
		if res_block_sep < 0 or res_block_sep > num_res_blocks:
			print("res block separation cannot be less than totoal res blocks", file=sys.stderr)

		deconv = AutoEncoder._get_deconv_type(input_type)
		
		upsample_layers_1 = [nn.ReLU(True),
						nn.BatchNorm2d(dim),
						deconv(dim, dim // 2, 4, 2, 1),]

		upsample_layers_2 = [nn.ReLU(True),
						nn.BatchNorm2d(dim // 2),
						deconv(dim // 2, input_dim, 4, 2, 1),]

		layers = []

		while num_res_blocks > res_block_sep:
			layers.append(ResBlock(dim, input_type))
			num_res_blocks -= 1

		layers += upsample_layers_1

		while num_res_blocks > 0:
			layers.append(ResBlock(dim // 2, input_type))
			num_res_blocks -= 1
		
		layers += upsample_layers_2
		
		return nn.Sequential(*layers)

class ResBlock(nn.Module):
	def __init__(self, dim, input_type):
		"""
			dim: dimension of volume to maintain
			input_type: data modality that encoder accepts image|audio
		"""
		super(ResBlock, self).__init__()
		conv = AutoEncoder._get_conv_type(input_type)
		self.block = nn.Sequential(
			conv(dim, dim, 3, 1, 1),
			nn.ReLU(True),
			nn.BatchNorm2d(dim),
			conv(dim, dim, 1),
			nn.ReLU(True),
			nn.BatchNorm2d(dim)
		)

	def forward(self, x):
		return x + self.block(x)

class PCIE(AutoEncoder):

	def __init__(self, img_res, input_dim, dim, perturb_feat=2, enc_type="shallow", dec_type="shallow", input_type="image"):
		super().__init__()
		self.encoder = AutoEncoder.get_encoder(input_dim, dim, enc_type)
		self.decoder = AutoEncoder.get_decoder(input_dim, dim+perturb_feat, dec_type)

		self.apply(self.weights_init)

		self.perturb_feat = perturb_feat

	def forward(self, x, perturb_feat_labels={}):

		z = self.encoder(x)
		
		x_tilde = self.decoder_forward(z, perturb_feat_labels)		
		x_tilde = torch.tanh(x_tilde)
		return x_tilde, z

	def decoder_forward(self, z, perturb_feat_labels={}):
		"""
			decoder forward needs separate default for perturb_feat_labels for
			interp based function where decoder forward is separately called.
		"""
		# print(perturb_feat_labels)
		# pdb.set_trace()
		if len(perturb_feat_labels) != self.perturb_feat:
			perturb_feat_labels = dict()

			# zero means no perturbation for the respective feature
			for feat in range(self.perturb_feat):
				perturb_feat_labels[f"perturb_feat_num_{feat}"] = torch.zeros(z.shape[0], device=z.device)
		
		# add perturb feature labels as extra activation maps
		# pdb.set_trace()
		labels = []
		for perturb_feat_name in perturb_feat_labels:
			
			label = perturb_feat_labels[perturb_feat_name].reshape(perturb_feat_labels[perturb_feat_name].shape[0], 1, 1, 1)
			
			label = label.expand_as(z)

			# cannot broadcast without broadcasting channel dim
			# so indexing it to reverse part of the broadcasting
			labels.append(label[:, :1, :, :])
		
		if len(perturb_feat_labels) > 0:
			labels = torch.cat(labels, dim=1)
			z = torch.cat([z, labels.float()], dim=1)



		x_tilde = self.decoder(z)
		return x_tilde

class ACAI(AutoEncoder):
	def __init__(self, img_res, input_dim, dim, enc_type="shallow", dec_type="shallow", input_type="image"):
		super().__init__()
		self.encoder = AutoEncoder.get_encoder(input_dim, dim, enc_type)
		self.decoder = AutoEncoder.get_decoder(input_dim, dim, dec_type)

		self.apply(self.weights_init)


	def forward(self, x):
		z = self.encoder(x)
		x_tilde = self.decoder(z)
		#x_tilde = torch.tanh(x_tilde)
		return x_tilde, z
	
	def decoder_forward(self, z):
		x_tilde = self.decoder(z)
		return x_tilde
	

class VAE(AutoEncoder):
	def __init__(self, input_dim, dim, enc_type="shallow", dec_type="shallow"):
		super(VAE, self).__init__()
		# mu and std each have size of dim
		self.encoder = AutoEncoder.get_encoder(input_dim, 2*dim, enc_type)
		self.decoder = AutoEncoder.get_decoder(input_dim, dim, dec_type)

		self.apply(self.weights_init)

	def forward(self, x):
		mu, logvar = self.encoder(x).chunk(2, dim=1)
		q_z_x = Normal(mu, logvar.mul(.5).exp())
		p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
		kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
		z = q_z_x.rsample()
		x_tilde = self.decoder(z)
		#x_tilde = torch.tanh(x_tilde)
		return x_tilde, z, kl_div

	def decoder_forward(self, z):
		x_tilde = self.decoder(z)
		return x_tilde

# class VAE(nn.Module):
#     def __init__(self, input_dim, dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_dim, dim, 4, 2, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 4, 2, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim, 5, 1, 0),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.Conv2d(dim, dim * 2, 3, 1, 0),
#             nn.BatchNorm2d(dim * 2)
#         )

#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(dim, dim, 3, 1, 0),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, dim, 5, 1, 0),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, dim, 4, 2, 1),
#             nn.BatchNorm2d(dim),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
#             nn.Tanh()
#         )

#         #self.apply(weights_init)

#     def forward(self, x):
#         mu, logvar = self.encoder(x).chunk(2, dim=1)

#         q_z_x = Normal(mu, logvar.mul(.5).exp())
#         p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
#         kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

#         x_tilde = self.decoder(q_z_x.rsample())
#         return x_tilde, kl_div

class VQEmbedding(nn.Module):
	def __init__(self, K, D):
		super(VQEmbedding, self).__init__()
		self.embedding = nn.Embedding(K, D)
		self.embedding.weight.data.uniform_(-1./K, 1./K)

	def forward(self, z_e_x):
		z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
		latents = vq(z_e_x_, self.embedding.weight)
		return latents

	def straight_through(self, z_e_x):
		z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
		z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
		z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

		z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
			dim=0, index=indices)
		z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
		z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

		return z_q_x, z_q_x_bar

class VectorQuantizedVAE(AutoEncoder):
	def __init__(self, input_dim, dim, K=512, enc_type="shallow", dec_type="shallow"):
		super(VectorQuantizedVAE, self).__init__()
		self.encoder = AutoEncoder.get_encoder(input_dim, dim, enc_type)

		self.codebook = VQEmbedding(K, dim)

		self.decoder = AutoEncoder.get_decoder(input_dim, dim, dec_type)

		self.apply(self.weights_init)

	def encode(self, x):
		z_e_x = self.encoder(x)
		latents = self.codebook(z_e_x)
		return latents

	def decode(self, latents):
		z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
		x_tilde = self.decoder(z_q_x)
		return x_tilde

	def forward(self, x):
		z_e_x = self.encoder(x)
		z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
		x_tilde = torch.tanh(self.decoder(z_q_x_st))
		return x_tilde, z_q_x, z_e_x,

	def decoder_forward(self, z):
		x_tilde = self.decoder(z)
		return x_tilde

class GatedActivation(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		x, y = x.chunk(2, dim=1)
		return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
	def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
		super().__init__()
		assert kernel % 2 == 1, print("Kernel size must be odd")
		self.mask_type = mask_type
		self.residual = residual

		self.class_cond_embedding = nn.Embedding(
			n_classes, 2 * dim
		)

		kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
		padding_shp = (kernel // 2, kernel // 2)
		self.vert_stack = nn.Conv2d(
			dim, dim * 2,
			kernel_shp, 1, padding_shp
		)

		self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

		kernel_shp = (1, kernel // 2 + 1)
		padding_shp = (0, kernel // 2)
		self.horiz_stack = nn.Conv2d(
			dim, dim * 2,
			kernel_shp, 1, padding_shp
		)

		self.horiz_resid = nn.Conv2d(dim, dim, 1)

		self.gate = GatedActivation()

	def make_causal(self):
		self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
		self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

	def forward(self, x_v, x_h, h):
		if self.mask_type == 'A':
			self.make_causal()

		h = self.class_cond_embedding(h)
		h_vert = self.vert_stack(x_v)
		h_vert = h_vert[:, :, :x_v.size(-1), :]
		out_v = self.gate(h_vert + h[:, :, None, None])

		h_horiz = self.horiz_stack(x_h)
		h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
		v2h = self.vert_to_horiz(h_vert)

		out = self.gate(v2h + h_horiz + h[:, :, None, None])
		if self.residual:
			out_h = self.horiz_resid(out) + x_h
		else:
			out_h = self.horiz_resid(out)

		return out_v, out_h


class GatedPixelCNN(nn.Module):
	def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
		super().__init__()
		self.dim = dim

		# Create embedding layer to embed input
		self.embedding = nn.Embedding(input_dim, dim)

		# Building the PixelCNN layer by layer
		self.layers = nn.ModuleList()

		# Initial block with Mask-A convolution
		# Rest with Mask-B convolutions
		for i in range(n_layers):
			mask_type = 'A' if i == 0 else 'B'
			kernel = 7 if i == 0 else 3
			residual = False if i == 0 else True

			self.layers.append(
				GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
			)

		# Add the output layer
		self.output_conv = nn.Sequential(
			nn.Conv2d(dim, 512, 1),
			nn.ReLU(True),
			nn.Conv2d(512, input_dim, 1)
		)

		self.apply(weights_init)

	def forward(self, x, label):
		shp = x.size() + (-1, )
		x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
		x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

		x_v, x_h = (x, x)
		for i, layer in enumerate(self.layers):
			x_v, x_h = layer(x_v, x_h, label)

		return self.output_conv(x_h)

	def generate(self, label, shape=(8, 8), batch_size=64):
		param = next(self.parameters())
		x = torch.zeros(
			(batch_size, *shape),
			dtype=torch.int64, device=param.device
		)

		for i in range(shape[0]):
			for j in range(shape[1]):
				logits = self.forward(x, label)
				probs = F.softmax(logits[:, :, i, j], -1)
				x.data[:, i, j].copy_(
					probs.multinomial(1).squeeze().data
				)
		return x