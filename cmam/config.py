# Absolut path of repository
base_path = '.'

# Place to store simulations
data_path ='./simulations'

# Template for jobscripts
jobscript_template = """#!/bin/bash -x
#SBATCH --job-name MAM_stim
#SBATCH -o {sim_dir}/{label}.%j.o
#SBATCH -e {sim_dir}/{label}.%j.e
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --cpus-per-task={local_num_threads}
#SBATCH --ntasks={num_processes}
#SBATCH --nodes={num_nodes} # -x jrc0650
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=j.pronold@fz-juelich.de
#SBATCH --account jinb33

module purge
module load GCC/10.3.0
module load CMake/3.18.0
module load ParaStationMPI/5.4.10-1
module load Python/3.8.5
module load SciPy-Stack/2021-Python-3.8.5
module load GSL/2.6
source {nest_dir}

srun python -u {base_path}/run_simulation.py {label} {network_label}"""

# Template for analysis jobscripts
jobscript_analysis_template = """#!/bin/bash -x
#SBATCH --job-name MAM_ana
#SBATCH -o {sim_dir}/{label}.%j.o
#SBATCH -e {sim_dir}/{label}.%j.e
#SBATCH --dependency=afterok:{jobid}
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task={local_num_threads_ana}
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mail-type=FAIL # notifications for job done & fail
#SBATCH --mail-user=j.pronold@fz-juelich.de
#SBATCH --account jinb33

module purge
module load GCC/10.3.0
module load CMake/3.18.0
module load ParaStationMPI/5.4.10-1
module load Python/3.8.5
module load SciPy-Stack/2021-Python-3.8.5
module load GSL/2.6
source {nest_dir}

srun python -u {base_path}/analysis.py {label} {network_label}"""
# Command to submit jobs on the local cluster
submit_cmd = 'sbatch' 
