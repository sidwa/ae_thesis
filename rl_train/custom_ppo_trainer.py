
from habitat import Config, logger
from habitat_baselines.rl.ppo import PPOTrainer, PPO
from custom_pointnav_policy import CustomPointNavPolicy, CustomPointNavNet

class CustomPPOTrainer(PPOTrainer):

	def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
		r"""Sets up actor critic and agent for PPO.

		Args:
			ppo_cfg: config node with relevant params

		Returns:
			None
		"""
		logger.add_filehandler(self.config.LOG_FILE)
		#print(ppo_cfg)

		# consolidate all net hyperparam in net_args
		# alternative would be to modify this method 
		# which would cause further modification in this class
		ppo_cfg.net_args.hidden_size = ppo_cfg.hidden_size
		self.actor_critic = PreTrainedPointNavPolicy(
			observation_space=self.envs.observation_spaces[0],
			action_space=self.envs.action_spaces[0],
			hidden_size=ppo_cfg.net_args,
			goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
		)
		self.actor_critic.to(self.device)

		self.agent = PPO(
			actor_critic=self.actor_critic,
			clip_param=ppo_cfg.clip_param,
			ppo_epoch=ppo_cfg.ppo_epoch,
			num_mini_batch=ppo_cfg.num_mini_batch,
			value_loss_coef=ppo_cfg.value_loss_coef,
			entropy_coef=ppo_cfg.entropy_coef,
			lr=ppo_cfg.lr,
			eps=ppo_cfg.eps,
			max_grad_norm=ppo_cfg.max_grad_norm,
			use_normalized_advantage=ppo_cfg.use_normalized_advantage,
		)