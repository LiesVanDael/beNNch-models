import sys
import os
from multiarea_model import MultiAreaModel
from start_jobs import start_job # create parameter folders
from chapter_experiments.q50_vanilla import network_params

num_processes = int(sys.argv[1])
local_num_threads = int(sys.argv[2])
t_sim = float(sys.argv[3])
rng_seed = int(sys.argv[4])
data_path = sys.argv[5]
label = sys.argv[6]

sim_params = {'num_processes': num_processes,
              'local_num_threads': local_num_threads,
              'recording_dict': {'record_vm': False},
              't_sim': t_sim,
              'rng_seed': rng_seed 
              }

if not record_spikes:
    sim_params['recording_dict']['areas_recorded'] = []

os.mkdir(os.path.join(data_path, label))

for net_params in network_params:
    M = MultiAreaModel(network_params,
                       simulation=True,
                       sim_spec=sim_params,
                       theory=False,
                       analysis=False)

print(M.label)
print(M.simulation.label)

start_job(M.simulation.label, data_path)
