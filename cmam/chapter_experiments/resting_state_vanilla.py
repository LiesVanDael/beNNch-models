import os
from config import base_path
from multiarea_model.default_params import nested_update, complete_area_list
from figures.Schmidt2018_dyn.network_simulations import NEW_SIM_PARAMS
from copy import deepcopy

# =============================================================================
# Extract network parameters from figure 3 and 5 from Schmidt et al
# =============================================================================

network_params_init3, _ = NEW_SIM_PARAMS['Fig3'][0]
network_params_init5, _ = NEW_SIM_PARAMS['Fig5'][0]
network_params_init3['connection_params']['K_stable'] = os.path.join(
    base_path, 'figures/SchueckerSchmidt2017/K_prime_original.npy'
)
network_params_init3['ana_params'] = {}
network_params_init3['ana_params']['chu2014_path'] = '/p/project/cjinb33/jinb3330/data/'
network_params_init5['connection_params']['K_stable'] = os.path.join(
    base_path, 'figures/SchueckerSchmidt2017/K_prime_original.npy'
)
network_params_init5['ana_params'] = {}
network_params_init5['ana_params']['chu2014_path'] = '/p/project/cjinb33/jinb3330/data/'

# =============================================================================
# Define simulation parameter dictionary
# =============================================================================

# Path to nest installation
nest = '/p/project/cjinb33/jinb3330/nest_installations/3_at_c690b7a_newer/bin/nest_vars.sh'
# JURECA DC has 128 cores on each node
vp_per_node = 128
# Each MPI process should have 4 threads
local_num_threads = 16
# 6 nodes on JURECA DC should work
num_nodes = 6
# Calculate number of tasks
num_processes = int(num_nodes * vp_per_node / local_num_threads)

sim_params = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 10000.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

sim_params_10 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 10500.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

sim_params_20 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 20000.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

sim_params_50 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 50500.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

sim_params_75 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 75500.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

sim_params_2 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 2000.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

sim_params_1 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 1000.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

sim_params_100 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 101500.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

sim_params_long = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 100000.0,
        'nest_dir': nest,
        'analysis': {'local_num_threads': 32}
        }

network_params = []

# =============================================================================
# Define list of different experiments to be carried out
# =============================================================================
cc_weights_factor = [1.9]
cc_weights_I_factor = [2.]
G = [-11.]

for g in G:
    for cc_scaling in cc_weights_factor:
        for cc_I_scaling in cc_weights_I_factor:
            if cc_scaling < 1.001:
                tmp = deepcopy(network_params_init3)
            else:
                tmp = deepcopy(network_params_init5)

            tmp = nested_update(
                    tmp,
                    {
                        'connection_params': {
                            'cluster_cc_connections': True,
                            'cc_weights_factor': cc_scaling,
                            'cc_weights_I_factor': cc_I_scaling,
                            'g': g
                            },
                        'ana_params': {'plotAllFiringRatesSummary': {'t_start': None, 't_stop': None}}
                    }
                    )
            network_params.append(tmp)
