import subprocess
import shlex
import pdb

# pdb.set_trace()
job_name = "cifar"

#command = f"sbatch -J {job_name} -o ./slurm/{job_name}.out -e ./slurm/{job_name}.err --parsable test_batch1.sh"
command = f"sbatch -J {job_name} -o ./slurm/lavda/{job_name}.out -e ./slurm/lavda/{job_name}.err --parsable sample_sbatch.sh 1024 10"
command = shlex.split(command)
res = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(res)

# print(res)
# jobid = int(res.stdout.decode(encoding="UTF-8").split("\n")[0])
# print(f"job id:{jobid}")

# job_name = "test2"
# command = f"sbatch --dependency=after:{jobid} +2 -J {job_name} -o ./slurm/{job_name}.out -e ./slurm/{job_name}.err test_batch2.sh"
# command = shlex.split(command)
# print(subprocess.run(command))
