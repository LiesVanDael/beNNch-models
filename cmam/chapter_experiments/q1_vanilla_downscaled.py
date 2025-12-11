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
# nest = '/p/project/cjinb33/vandael1/builds/3.8_cMAM_update/install/bin/nest_vars.sh'
# JURECA DC has 128 cores on each node
vp_per_node = 16
# Each MPI process should have 4 threads
local_num_threads = 1
# 6 nodes on JURECA DC should work
num_nodes = 1 
# Calculate number of tasks
num_processes = int(num_nodes * vp_per_node / local_num_threads)

sim_params_1 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 1000.0,
        'analysis': {'local_num_threads': 16}
        }

sim_params_2 = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 2000.0,
        'analysis': {'local_num_threads': 16}
        }

network_params = []

# =============================================================================
# Define list of different experiments to be carried out
# =============================================================================
cc_weights_factor = [2.]  # [2.5]  # [2.]  # [1.0]
cc_weights_I_factor = [2.2]  # [2.1, 2.2]  # [2.1, 2.2, 2.3, 2.4]  # [2.5, 3.]
Q = 1
J_E_PLUS = [1]  # [9, 10, 11]  # [8]  # [7]  # [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20.]
RJ = [.75]
G = [-11.]  # [-10., -9.]

for g in G:
    for rj in RJ:
        for cc_scaling in cc_weights_factor:
            for cc_I_scaling in cc_weights_I_factor:
                for j_e_plus in J_E_PLUS:
                    if cc_scaling < 1.001:
                        tmp = deepcopy(network_params_init3)
                    else:
                        tmp = deepcopy(network_params_init5)
                    cluster_stimulation_default_params = {
                            # Start of the stimulating input (in ms).
                            'stim_start': 600.0,
                            # Duration of the stimulating input (in ms).
                            'stim_duration': 100.0,
                            # Rate of stimulation
                            'stim_rate': 0.,
                            }
                    all_cluster_stimulation_default_params = {str(q): deepcopy(cluster_stimulation_default_params) for q in range(int(Q))}

                    pulvinar_stimulation_default_params = {
                            # Start of the stimulating input (in ms).
                            'stim_start': 600.0,
                            # Duration of the stimulating input (in ms).
                            'stim_duration': 100.0,
                            # Rate of stimulation
                            'stim_rate': 0.,
                            }
                    all_pulvinar_stimulation_default_params = {str(q): deepcopy(pulvinar_stimulation_default_params) for q in range(int(Q))}

                    cluster_default_params = {
                            'Q': int(Q),
                            'J_E_plus': j_e_plus,
                            'R_J': rj,
                            # 3.3 % of al connections are from non-visual and subcortical structures,
                            # 28,5 % of all connections are from within a patch. 3.3 + 28.5 = 31.8 of
                            # all connections are not from within the model. Here we assume that all
                            # non-visual/subcortical neurons stem from the pulvinar.
                            # pulvinar connections should be roughly 3.3 / (28.5 + 3.3) * K_ext
                            'pulvinar_frac_type_4_vs_3': 1.,
                            'cluster_stimulation_parameters': all_cluster_stimulation_default_params,
                            'pulvinar_stimulation_parameters': all_pulvinar_stimulation_default_params
                            }
                    clusters = {area: deepcopy(cluster_default_params) for area in complete_area_list}

                    tmp = nested_update(
                            tmp,
                            {
                                'connection_params': {
                                    'cluster_cc_connections': True,
                                    'cc_weights_factor': cc_scaling,
                                    'cc_weights_I_factor': cc_I_scaling,
                                    'Q': Q,
                                    'cc_J_E_plus': j_e_plus,
                                    'cc_R_J': rj,
                                    'g': g
                                    },
                                'stim_params': {'frac_type_4_vs_3': 1., 'cluster_frac_type_4_vs_3': 1.},
                                'cluster': clusters,
                                'ana_params': {'plotAllFiringRatesSummary': {'t_start': None, 't_stop': None}}
                            }
                            )
                    network_params.append(tmp)
