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

sim_params_long = {
        'num_processes': num_processes,
        'local_num_threads': local_num_threads,
        'num_nodes': num_nodes,
        'recording_dict': {'record_vm': False},
        't_sim': 100000.0,
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

network_params = []

# =============================================================================
# Define list of different experiments to be carried out
# =============================================================================
cc_weights_factor = [2.]  # [2.5]  # [2.]  # [1.0]
cc_weights_I_factor = [2.2]  # [2.1, 2.2]  # [2.1, 2.2, 2.3, 2.4]  # [2.5, 3.]
Q = 50
J_E_PLUS = [15]  # [9, 10, 11]  # [8]  # [7]  # [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20.]
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

                    # Ignition experiment

                    dt_init = 500.  # Time for initial transients, in ms
                    dt_trial = 1000.
                    # T = 100500
                    number_of_rounds = 100  # 1 + int((T-dt_init) / dt_trial) # Assumes simulation time T = 100500 ms

                    stim_rates_1 = {str(q): [] for q in range(int(Q))}
                    stim_starts_1 = {str(q): [] for q in range(int(Q))}

                    # 500, 3000, 5500, 8000, 10500, 13000
                    # If stim_magnitude is too high the cue is inhibited
                    # Either stim_magnitude = 2., cue_magnitude = 3.
                    # Either stim_magnitude = 1., cue_magnitude = 4.
                    stim_magnitude = [
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            30., 30., 30., 30., 30., 30., 30., 30., 30., 30.,
                            ]

                    stim_duration = [
                            50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
                            50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
                            50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
                            50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
                            50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
                            200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                            200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                            200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                            200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                            200., 200., 200., 200., 200., 200., 200., 200., 200., 200.,
                            ]

                    for i in range(number_of_rounds):
                        q_1 = str(i%int(Q))

                        stim_on = dt_init + i * dt_trial
                        stim_end = stim_on + stim_duration[i]

                        stim_starts_1[q_1].append(stim_on)
                        stim_starts_1[q_1].append(stim_end)
                        stim_rates_1[q_1].append(stim_magnitude[i])
                        stim_rates_1[q_1].append(0.)

                    for q in stim_starts_1.keys():
                        s = stim_starts_1[q]
                        r = stim_rates_1[q]

                        clusters['V1']['cluster_stimulation_parameters'][q]['stim_start'] = s
                        clusters['V1']['cluster_stimulation_parameters'][q]['stim_rate'] = r

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
