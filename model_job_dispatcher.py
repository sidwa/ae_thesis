import subprocess
import shlex
import pickle
import argparse
import os

from model_summary import get_batch_size

def sbatch_loss_job(args):
	
	# calculated batch size can be more than available batch size
	# so batch size is reduced by the factor below.
	if args.model == "acai":
		correction_factor = 20
	elif args.model == "vqvae":
		correction_factor = 10
	elif args.model == "vae":
		correction_factor = 9

	with open(f"model_batch_size_{args.server_type}", "rb") as bs:
		batch_dict = pickle.load(bs)

	#hidden_sizes = [256, 128, 64, 48, 32]
	losses = ["mse", "gan", "comp","comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc"]
	# losses = ["mse"]
	#for hidden_size in hidden_sizes:
	for args.recons_loss in losses:

		if args.recons_loss != "mse":
			recons_loss_type = "gan"
		else:
			recons_loss_type = "mse"
		hidden_size = 128
		batch_size = int(batch_dict[f"{args.model}_{args.enc_type}_{args.dec_type}_{hidden_size}_{recons_loss_type}"])*args.num_gpu
		batch_size -= (correction_factor * args.num_gpu)
	
		batch_size = max(batch_size, 1)

		# if args.model == "acai":
		# 	correction_factor += 7
		# if args.model == "vqvae":
		# 	correction_factor += 10
		# if args.model == "vae":
		# 	correction_factor += 9
		

		job_name = f"{args.model}_{args.enc_type}_{args.dec_type}_{hidden_size}_{args.recons_loss}"
		log_dir = os.path.join(".","slurm",args.recons_loss)
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		log_path = os.path.join(log_dir, job_name)
		#orig_command = ["sbatch", "-J", job_name, "-o", f"./slurm/{job_name}.out", "-e", f"./slurm/{job_name}.err", f"{args.model}_job_sbatch.sh", args.enc_type, args.dec_type, str(hidden_size), str(batch_size), args.recons_loss]
		command = f"sbatch -J {job_name} -o {log_path}.out -e {log_path}.err --parsable {args.model}_job_sbatch.sh init \
			 		{args.enc_type} {args.dec_type} {str(hidden_size)} {str(batch_size)} {args.recons_loss} {args.dataset} \
					{args.img_res} {args.num_epochs}"
		command = shlex.split(command)
		res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		jobid = int(res.stdout.decode(encoding="UTF-8").split("\n")[0])
		print(res)
		# continue_train_command = f"sbatch --dependency=afternotok:{jobid} -J {job_name}_cont -o {log_path}_cont.out \ 
		# 		-e {log_path}_cont.err {args.model}_job_sbatch.sh load {args.enc_type} {args.dec_type} {str(hidden_size)} \
		# 		{str(batch_size)} {args.recons_loss} {args.dataset} {args.img_res} {args.num_epochs}"
		# continue_train_command = shlex.split(continue_train_command)
		# print(subprocess.run(continue_train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE))


def sbatch_job(args):
	
	# calculated batch size can be more than available batch size
	# so batch size is reduced by the factor below.
	if args.model == "acai":
		correction_factor = 20
	elif args.model == "vqvae":
		correction_factor = 10
	elif args.model == "vae":
		correction_factor = 9

	with open(f"model_batch_size_{args.server_type}", "rb") as bs:
		batch_dict = pickle.load(bs)

	# XXXXXXXX TEMPORARY!!! XXXXXXXXXXXXXXXXX
	if args.recons_loss != "mse":
		recons_loss_type = "gan"
	else:
		recons_loss_type = "mse"
	# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

	hidden_sizes = [256, 128, 64, 48, 32]
	# hidden_sizes = [128]
	# losses = ["comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc"]
	for hidden_size in hidden_sizes:
	#for args.recons_loss in losses:
		hidden_size = 128
		batch_size = int(batch_dict[f"{args.model}_{args.enc_type}_{args.dec_type}_{hidden_size}_{recons_loss_type}"])*args.num_gpu
		batch_size -= (correction_factor * args.num_gpu)
		
		if args.model == "acai":
			correction_factor += 7
		if args.model == "vqvae":
			correction_factor += 10
		if args.model == "vae":
			correction_factor += 9
		

		job_name = f"{args.model}_{args.enc_type}_{args.dec_type}_{hidden_size}_{args.recons_loss}"
		log_dir = os.path.join(".","slurm",args.recons_loss)
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		log_path = os.path.join(log_dir, job_name)
		#orig_command = ["sbatch", "-J", job_name, "-o", f"./slurm/{job_name}.out", "-e", f"./slurm/{job_name}.err", f"{args.model}_job_sbatch.sh", args.enc_type, args.dec_type, str(hidden_size), str(batch_size), args.recons_loss]
		command = f"sbatch -J {job_name} -o {log_path}.out -e {log_path}.err --parsable {args.model}_job_sbatch.sh init \
			 		{args.enc_type} {args.dec_type} {str(hidden_size)} {str(batch_size)} {args.recons_loss} {args.dataset} \
					{args.img_res} {args.num_epochs}"
		command = shlex.split(command)
		res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		jobid = int(res.stdout.decode(encoding="UTF-8").split("\n")[0])

		# continue_train_command = f"sbatch --dependency=afternotok:{jobid} -J {job_name}_cont -o {log_path}_cont.out \ 
		# 		-e {log_path}_cont.err {args.model}_job_sbatch.sh load {args.enc_type} {args.dec_type} {str(hidden_size)} \
		# 		{str(batch_size)} {args.recons_loss} {args.dataset} {args.img_res} {args.num_epochs}"
		# continue_train_command = shlex.split(continue_train_command)
		# print(subprocess.run(continue_train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

def print_batch_size(args):
	with open(f"model_batch_size_{args.server_type}", "rb") as bs:
		batch_dict = pickle.load(bs)

	if args.model == "acai":
		correction_factor = 20
	elif args.model == "vqvae":
		correction_factor = 10
	elif args.model == "vae":
		correction_factor = 0

	hidden_sizes = [256, 128, 64, 48, 32]
	for hidden_size in hidden_sizes:

		
		batch_size = int(batch_dict[f"{args.model}_{args.enc_type}_{args.dec_type}_{hidden_size}_{args.recons_loss}"])*args.num_gpu
		batch_size -= correction_factor
		
		if args.model == "acai":
			correction_factor += 7
		if args.model == "vqvae":
			correction_factor += 10
		if args.model == "vae":
			correction_factor += 9

		print(f"model:{args.model} hs:{hidden_size} :: {batch_size}")


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("mode", type=str, choices={"batch_size", "job", "loss_func"})
	parser.add_argument("--dataset", type=str, choices={"imagenet", "tiny-imagenet"})
	parser.add_argument("--img_res", type=int, default=128)
	parser.add_argument("--model", type=str, choices={"vqvae", "vae", "acai"})
	parser.add_argument("--enc_type", type=str, default="shallow")
	parser.add_argument("--dec_type", type=str, default="shallow")
	# parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--recons_loss", type=str, choices={"mse", "gan", "comp", "comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc"})
	parser.add_argument("--num_epochs", default=20, type=int)
	parser.add_argument("--server_type", type=str, choices={"rc", "cuda", "v100"})
	parser.add_argument("--num_gpu", type=int)

	args = parser.parse_args()

	if args.mode == "batch_size":
		print_batch_size(args)
	elif args.mode == "job":
		sbatch_job(args)
	elif args.mode == "loss_func":
		sbatch_loss_job(args)

