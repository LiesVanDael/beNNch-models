import numpy as np
import sys
import os
from multiarea_model import MultiAreaModel
from start_jobs import start_job # create parameter folders
from params import params

num_processes = int(sys.argv[1])
local_num_threads = int(sys.argv[2])
t_sim = float(sys.argv[3])
rng_seed = int(sys.argv[4])
data_path = sys.argv[5]
label = sys.argv[6]
record_spikes = sys.argv[7].lower() in ("true", "1", "yes")
Q = int(sys.argv[8])
poisson_input = sys.argv[9].lower() in ("true", "1", "yes")

sim_params = {'num_processes': num_processes,
              'local_num_threads': local_num_threads,
              'recording_dict': {'record_vm': False},
              't_sim': t_sim,
              'rng_seed': rng_seed 
              }

if not record_spikes:
    sim_params['recording_dict']['areas_recorded'] = []

os.makedirs(os.path.join(data_path, label), exist_ok=True)

network_params = params(Q)

for net_params in network_params:
    network_params['input_params']['poisson_input'] = poisson_input
    M = MultiAreaModel(net_params,
                       simulation=True,
                       sim_spec=sim_params,
                       theory=False,
                       analysis=False,
                       data_path=data_path,
                       data_folder_hash=label)

print("M.label: ", M.label)
print("M.simulation.label: ", M.simulation.label)

start_job(M.simulation.label, data_path, data_folder_hash=label)
