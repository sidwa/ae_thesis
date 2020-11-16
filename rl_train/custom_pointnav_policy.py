
import torch.nn as nn
import torch
from habitat_baselines.rl.ppo.policy import Policy, PointNavBaselineNet
from .. import modules


class VisualEncoder(nn.Module):
	"""
		Custom network to test if any custom network can be made to work for habitat_baseline
	"""

	def __init__(self, observation_space, net_args):
		if "rgb" in observation_space.spaces:
			self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
		else:
			self._n_input_rgb = 0

		if "depth" in observation_space.spaces:
			self._n_input_depth = observation_space.spaces["depth"].shape[2]
		else:
			self._n_input_depth = 0

		obs_h_w = self._get_observation_dims(observation_space)

		# Meant to be a network
		self.ae_encoder = modules.get_encoder(self._n_input_rgb, net_args.ae_hidden_size)

		if self._n_input_depth > 0:
			# depth network is not pre trained as trained with ppo agent
			# input :: h * w * _n_input_rgb
			# output:: h//4 * w//4 * depth_hidden_size
			self.depth_encoder = self._get_depth_net(self._n_input_depth, net_args.depth_hidden_size)
								
		# uses more abstract features from cnn_raw as input ae_encoder is placeholder for encoder from an AE
		self.encoder = nn.Sequential(nn.Conv2d(net_args.ae_hidden_size+net_args.depth_hidden_size, 128, 8, 1),
											nn.ReLU(True),
											nn.Conv2d(128, 64, 4, 2),
											nn.ReLU(True),
											nn.Conv2d(64, 32, 3, 1),
											nn.ReLU(True),
											nn.Flatten(),
											nn.Linear(32 * obs_h_w[0] // 8 * obs_h_w[1] // 8, net_args.hidden_size),
											nn.ReLU(True),
											)

		self.layer_init()

	def layer_init(self):
		nets = [self.ae_encoder, self.depth_encoder, self.visual_encoder]
		for net in nets:
			for layer in net:
				if isinstance(layer, (nn.Conv2d, nn.Linear)):
					nn.init.kaiming_normal_(
						layer.weight, nn.init.calculate_gain("relu")
					)
					if layer.bias is not None:
						nn.init.constant_(layer.bias, val=0)

	def _get_observation_dims(self, observation_space):
		"""
			gets observation height and width
		"""
		if self._n_input_rgb > 0:
			obs_dims = np.array(
				observation_space.spaces["rgb"].shape[:2], dtype=np.float32
			)
		elif self._n_input_depth > 0:
			obs_dims = np.array(
				observation_space.spaces["depth"].shape[:2], dtype=np.float32
			)
		
		return obs_dims

	def _get_depth_net(self, input_depth, depth_hidden_size):

		net = nn.Sequential(nn.Conv2d(input_depth, 32, 7, 2),
							nn.ReLU(True),
							nn.Conv2d(16, 64, 5, 1),
							nn.ReLU(True),
							nn.Conv2d(32, 128, 3, 2),
							nn.ReLU(True),
							nn.Conv2d(64, depth_hidden_size, 3, 1),
							nn.ReLU(True)
							)

		return net


	def forward(self, observations):
		if self._n_input_rgb > 0:
			rgb_observations = observations["rgb"]
			
			# permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
			rgb_observations = rgb_observations.permute(0, 3, 1, 2)
			
			#with torch.no_grad:
			rgb_out = self.ae_encoder(rgb_observations)
			
		if self._n_input_depth > 0:
			depth_observations = observations["depth"]
			
			# permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
			depth_observations = depth_observations.permute(0, 3, 1, 2)
			depth_out = self.depth_encoder(depth_observations)

		encoder_input = torch.cat([rgb_out, depth_out], dim=1)

		return self.encoder(encoder_input)



class CustomPointNavNet(PointNavBaselineNet):
	"""
		encapsulate Custom network to test if any custom network can be made to work for habitat_baseline
	"""
	def __init__(self, observation_space, hidden_size, goal_sensor_uuid):
		super.__init__(self, observation_space, hidden_size, goal_sensor_uuid)

		self.visual_encoder = self.get_visual_encoder(observation_space, hidden_size)

	def get_visual_encoder(self, observation_space, hidden_size):
		if "rgb" in observation_space.spaces:
			self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
		else:
			self._n_input_rgb = 0

		if "depth" in observation_space.spaces:
			self._n_input_depth = observation_space.spaces["depth"].shape[2]
		else:
			self._n_input_depth = 0

		obs_h_w = self._get_observation_dims(observation_space)

		# extracts features from raw observation
		self.ae_encoder = modules.get_encoder(4, 256)

		# uses more abstract features from cnn_raw as input ae_encoder is placeholder for encoder from an AE
		self.visual_encoder = nn.Sequential(self.ae_encoder,
											nn.Conv2d(256, 128, 8, 1),
											nn.ReLU(True),
											nn.Conv2d(128, 64, 4, 2),
											nn.ReLU(True),
											nn.Conv2d(64, 32, 3, 1),
											nn.ReLU(True),
											nn.Flatten(),
											nn.Linear(32 * obs_h_w[0] // 8 * obs_h_w[1] // 8, hidden_size),
											nn.ReLU(True),
											)

	def _get_observation_dims(self, observation_space):
		"""
			gets observation height and width
		"""
		if self._n_input_rgb > 0:
			obs_dims = np.array(
				observation_space.spaces["rgb"].shape[:2], dtype=np.float32
			)
		elif self._n_input_depth > 0:
			obs_dims = np.array(
				observation_space.spaces["depth"].shape[:2], dtype=np.float32
			)
		
		return obs_dims

class PreTrainedPointNavNet(PointNavBaselineNet):
	"""
		Overrides the visual encoder of baseline net with a pretrained net
		the pretrained net would be an encoder  
	"""
	def __init__(self, observation_space, net_args, goal_sensor_uuid):
		super.__init__(self, observation_space, net_args.hidden_size, goal_sensor_uuid)
		
		self.visual_encoder = VisualEncoder(observation_space, net_args)

	# def set_visual_encoder(self, observation_space, net_args):
	# 	if "rgb" in observation_space.spaces:
	# 		self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
	# 	else:
	# 		self._n_input_rgb = 0

	# 	if "depth" in observation_space.spaces:
	# 		self._n_input_depth = observation_space.spaces["depth"].shape[2]
	# 	else:
	# 		self._n_input_depth = 0

	# 	obs_h_w = self._get_observation_dims(observation_space)

	# 	# extracts features from raw observation
	# 	# input :: h * w * _n_input_rgb
	# 	# output:: h//4 * w//4 * ae_hidden_size
	# 	self.ae_encoder = modules.get_encoder(self._n_input_rgb, net_args.ae_hidden_size)

	# 	if self._n_input_depth > 0:
	# 		# depth network is not pre trained as trained with ppo agent
	# 		# input :: h * w * _n_input_rgb
	# 		# output:: h//4 * w//4 * depth_hidden_size
	# 		self.depth_encoder = self._get_depth_net(self._n_input_depth, net_args.depth_hidden_size)
								
	# 	# uses more abstract features from cnn_raw as input ae_encoder is placeholder for encoder from an AE
	# 	self.visual_encoder = nn.Sequential(nn.Conv2d(net_args.ae_hidden_size+net_args.depth_hidden_size, 128, 8, 1),
	# 										nn.ReLU(True),
	# 										nn.Conv2d(128, 64, 4, 2),
	# 										nn.ReLU(True),
	# 										nn.Conv2d(64, 32, 3, 1),
	# 										nn.ReLU(True),
	# 										nn.Flatten(),
	# 										nn.Linear(32 * obs_h_w[0] // 8 * obs_h_w[1] // 8, net_args.hidden_size),
	# 										nn.ReLU(True),
	# 										)
		
	
class PreTrainedPointNavPolicy(Policy):
	def __init__(
		self,
		observation_space,
		action_space,
		goal_sensor_uuid,
		net_args,
	):
		super().__init__(
			PreTrainedPointNavNet(
				observation_space=observation_space,
				net_args=net_args,
				goal_sensor_uuid=goal_sensor_uuid,
			),
			action_space.n,
		)
	

class CustomPointNavPolicy(Policy):
	def __init__(
		self,
		observation_space,
		action_space,
		goal_sensor_uuid,
		hidden_size=512,
	):
		super().__init__(
			CustomPointNavNet(
				observation_space=observation_space,
				hidden_size=hidden_size,
				goal_sensor_uuid=goal_sensor_uuid,
			),
			action_space.n,
		)