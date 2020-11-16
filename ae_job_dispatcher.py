import subprocess
import shlex
import pickle
import argparse
import os

def sbatch_loss_job(args):
	
	# calculated batch size can be more than available batch size
	# so batch size is reduced by the factor below.
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
		

		job_name = f"ae_{args.enc_type}_{args.dec_type}_{hidden_size}_{args.recons_loss}"
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


def sbatch_model_job(args):
	
	# calculated batch size can be more than available batch size
	# so batch size is reduced by the factor below.
	
	#hidden_sizes = [256, 128, 64, 48, 32]
	losses = ["mse", "comp_6_adv", "comp_6_dc"]
	# losses = ["mse"]

	enc_size = ["shallow", "moderate_shallow", "moderate"]
	dec_size = ["shallow", "moderate_shallow", "moderate"]

	args.recons_loss = "mse"
	#for hidden_size in hidden_sizes:
	for prior_loss in ["gan"]:

		for perturbs in [""]:

			for perturb_gan in [""]:

				for recons_loss in losses:

					for args.enc_type in enc_size:
						for args.dec_type in dec_size:

							if args.enc_type =="moderate_shallow" and args.dec_type != "moderate":
								continue

							if perturb_gan == "--perturb_feat_gan":
								pert_gan = "yes"
								# args.num_epochs += 15
							else:
								pert_gan = "no"

							if args.recons_loss != "mse":
								recons_loss_type = "gan"
							else:
								recons_loss_type = "mse"
							hidden_size = 128
						
							batch_size = 64 * args.num_gpu

							if args.enc_type == "shallow_moderate" or args.dec_type == "shallow_moderate":
								batch_size //= 2
							
							if args.enc_type == "moderate" or args.dec_type == "moderate":
								batch_size //= 4
							
							pert_names = "rot+blur" if perturbs == "--rotate_perturb --blur_perturb" else perturbs
							job_name = f"ae_pr{prior_loss}_pert_gan_{pert_gan}_pert_{pert_names}_enc_{args.enc_type}_dec_{args.dec_type}"
							log_dir = os.path.join(".","slurm",f"{args.recons_loss}_{args.dataset}")
							if not os.path.exists(log_dir):
								os.makedirs(log_dir)
							log_path = os.path.join(log_dir, job_name)
							#orig_command = ["sbatch", "-J", job_name, "-o", f"./slurm/{job_name}.out", "-e", f"./slurm/{job_name}.err", f"{args.model}_job_sbatch.sh", args.enc_type, args.dec_type, str(hidden_size), str(batch_size), args.recons_loss]
							command = f"sbatch -J {job_name} -o {log_path}.out -e {log_path}.err -t {args.time} --parsable ae_job_sbatch.sh init \
										{args.enc_type} {args.dec_type} {str(hidden_size)} {str(batch_size)} {recons_loss} {args.dataset} \
										{args.img_res} {args.num_epochs} {prior_loss} {perturbs} {perturb_gan}"


							
							command = shlex.split(command)
							res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
							# jobid = int(res.stdout.decode(encoding="UTF-8").split("\n")[0])
							# print(res)
							# continue_train_command = f"sbatch --dependency=afternotok:{jobid} -J {job_name}_cont -o {log_path}_cont.out \ 
							# 		-e {log_path}_cont.err {args.model}_job_sbatch.sh load {args.enc_type} {args.dec_type} {str(hidden_size)} \
							# 		{str(batch_size)} {args.recons_loss} {args.dataset} {args.img_res} {args.num_epochs}"
							# continue_train_command = shlex.split(continue_train_command)
							# print(subprocess.run(continue_train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
	
def perturb_job(args):
	
	# calculated batch size can be more than available batch size
	# so batch size is reduced by the factor below.
	
	#hidden_sizes = [256, 128, 64, 48, 32]
	losses = ["mse", "comp_6_adv", "comp_6_dc"]
	# losses = ["mse"]

	args.recons_loss = "mse"
	#for hidden_size in hidden_sizes:
	for prior_loss in ["kl_div", "gan"]:

		for perturbs in ["", "--rotate_perturb", "--blur_perturb", "--rotate_perturb --blur_perturb"]:

			for perturb_gan in ["", "--perturb_feat_gan"]:

				for recons_loss in losses:

					if perturb_gan == "--perturb_feat_gan":
						pert_gan = "yes"
						args.num_epochs += 5
					else:
						pert_gan = "no"

					hidden_size = 128
				
					batch_size = 64 * args.num_gpu
					pert_names = "rot+blur" if perturbs == "--rotate_perturb --blur_perturb" else perturbs
					job_name = f"ae_pr{prior_loss}_pert_gan_{pert_gan}_pert_{pert_names}_loss_{recons_loss}"
					log_dir = os.path.join(".","slurm",f"{args.recons_loss}_{args.dataset}")
					if not os.path.exists(log_dir):
						os.makedirs(log_dir)
					log_path = os.path.join(log_dir, job_name)
					#orig_command = ["sbatch", "-J", job_name, "-o", f"./slurm/{job_name}.out", "-e", f"./slurm/{job_name}.err", f"{args.model}_job_sbatch.sh", args.enc_type, args.dec_type, str(hidden_size), str(batch_size), args.recons_loss]
					command = f"sbatch -J {job_name} -o {log_path}.out -e {log_path}.err -t {args.time} --parsable ae_job_sbatch.sh init \
								{args.enc_type} {args.dec_type} {str(hidden_size)} {str(batch_size)} {recons_loss} {args.dataset} \
								{args.img_res} {args.num_epochs} {prior_loss} {perturbs} {perturb_gan}"


					
					command = shlex.split(command)
					res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					# jobid = int(res.stdout.decode(encoding="UTF-8").split("\n")[0])
					# print(res)
					# continue_train_command = f"sbatch --dependency=afternotok:{jobid} -J {job_name}_cont -o {log_path}_cont.out \ 
					# 		-e {log_path}_cont.err {args.model}_job_sbatch.sh load {args.enc_type} {args.dec_type} {str(hidden_size)} \
					# 		{str(batch_size)} {args.recons_loss} {args.dataset} {args.img_res} {args.num_epochs}"
					# continue_train_command = shlex.split(continue_train_command)
					# print(subprocess.run(continue_train_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument("mode", type=str, choices={"perturb", "model_size", "loss_func"})
	parser.add_argument("--dataset", type=str, choices={"imagenet", "tiny-imagenet"})
	parser.add_argument("--img_res", type=int, default=128)
	parser.add_argument("--enc_type", type=str, default="shallow")
	parser.add_argument("--dec_type", type=str, default="shallow")
	# parser.add_argument("--hidden_size", type=int, default=256)
	parser.add_argument("--recons_loss", type=str, choices={"mse", "gan", "comp", "comp_2_adv", "comp_2_dc", "comp_6_adv", "comp_6_dc"})
	parser.add_argument("--num_epochs", default=20, type=int)
	parser.add_argument("--server_type", type=str, choices={"rc", "cuda", "v100"})
	parser.add_argument("--time", type=str, default="5-0:0:0")
	parser.add_argument("--num_gpu", type=int)

	args = parser.parse_args()

	if args.mode == "perturb":
		perturb_job(args)
	elif args.mode == "model_size":
		sbatch_model_job(args)
	elif args.mode == "loss_func":
		sbatch_loss_job(args)

