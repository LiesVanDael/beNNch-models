"""
default_parameters.py
=====================
This script defines the default values of all
parameters and defines functions to compute
single neuron and synapse parameters and to
properly set the seed of the random generators.

Authors
-------
Maximilian Schmidt
"""

from config import base_path
from copy import deepcopy
import json
import os
import nest

import numpy as np

complete_area_list = ['V1', 'V2', 'VP', 'V3', 'V3A', 'MT', 'V4t', 'V4', 'VOT', 'MSTd',
                      'PIP', 'PO', 'DP', 'MIP', 'MDP', 'VIP', 'LIP', 'PITv', 'PITd',
                      'MSTl', 'CITv', 'CITd', 'FEF', 'TF', 'AITv', 'FST', '7a', 'STPp',
                      'STPa', '46', 'AITd', 'TH']

population_list = ['23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']

f1 = open(os.path.join(base_path, 'multiarea_model/data_multiarea',
                       'viscortex_raw_data.json'), 'r')
raw_data = json.load(f1)
f1.close()
av_indegree_Cragg = raw_data['av_indegree_Cragg']
av_indegree_OKusky = raw_data['av_indegree_OKusky']


"""
Simulation parameters
"""
sim_params = {
    # rng seed for random number generators
    'rng_seed': 7,
    # simulation step (in ms)
    'dt': 0.1,
    # simulated time (in ms)
    't_sim': 10.0,
    # no. of MPI processes:
    'num_processes': 1,
    # no. of nodes:
    'num_nodes': 1,
    # no. of threads per MPI process':
    'local_num_threads': 1,
    # Set nest direction
    'nest_dir': None,
    # Areas represented in the network
    'areas_simulated': complete_area_list,
    # Parameters to use in the analysis
    'analysis': {
        # Number of threads to us in the analysis
        'local_num_threads': 1
        }
}

"""
Network parameters
"""
network_params = {
    # Surface area of each area in mm^2
    'surface': 1.0,
    # Scaling of population sizes
    'N_scaling': 1.,
    # Scaling of indegrees
    'K_scaling': 1.,
    # Absolute path to the file holding full-scale rates for scaling
    # synaptic weights
    'fullscale_rates': None,
    # Check whether NEST 2 or 3 is used. No straight way of checking this is
    # available. But PrintNetwork was removed in NEST 3, so checking for its
    # existence should suffice.
    'USING_NEST_3': 'PrintNetwork' not in dir(nest)
}


"""
Single-neuron parameters
"""

sim_params.update(
    {
        'initial_state': {
            # mean of initial membrane potential (in mV)
            'V_m_mean': -58.0,
            # std of initial membrane potential (in mV)
            'V_m_std': 10.0
        }
    })

# dictionary defining single-cell parameters
single_neuron_dict = {
    # Leak potential of the neurons (in mV).
    'E_L': -65.0,
    # Threshold potential of the neurons (in mV).
    'V_th': -50.0,
    # Membrane potential after a spike (in mV).
    'V_reset': -65.0,
    # Membrane capacitance (in pF).
    'C_m': 250.0,
    # Membrane time constant (in ms).
    'tau_m': 10.0,
    # Time constant of postsynaptic excitatory currents (in ms).
    'tau_syn_ex': 0.5,
    # Time constant of postsynaptic inhibitory currents (in ms).
    'tau_syn_in': 0.5,
    # Refractory period of the neurons after a spike (in ms).
    't_ref': 2.0}

neuron_params = {
    # neuron model
    'neuron_model': 'iaf_psc_exp',
    # neuron parameters
    'single_neuron_dict': single_neuron_dict,
    # Mean and standard deviation for the
    # distribution of initial membrane potentials
    'V0_mean': -100.,
    'V0_sd': 50.}

network_params.update({'neuron_params': neuron_params})


"""
General connection parameters
"""
connection_params = {
    # Whether to apply the stabilization method of
    # Schuecker, Schmidt et al. (2017). Default is False.
    # Options are True to perform the stabilization or
    # a string that specifies the name of a binary
    # numpy file containing the connectivity matrix
    'K_stable': None,

    # Whether to replace all cortico-cortical connections by stationary
    # Poisson input with population-specific rates (het_poisson_stat)
    # or by time-varying current input (het_current_nonstat)
    # while still simulating all areas. In both cases, the data to replace
    # the cortico-cortical input is loaded from `replace_cc_input_source`.
    'replace_cc': False,

    # Whether to replace non-simulated areas by Poisson sources
    # with the same global rate rate_ext ('hom_poisson_stat') or
    # by specific rates ('het_poisson_stat')
    # or by time-varying specific current ('het_current_nonstat')
    # In the two latter cases, the data to replace the cortico-cortical
    # input is loaded from `replace_cc_input_source`
    'replace_non_simulated_areas': None,

    # Source of the input rates to replace cortico-cortical input
    # Either a json file (has to end on .json) holding a scalar values
    # for each population or
    # a base name such that files with names
    # $(replace_cc_input_source)-area-population.npy
    # (e.g. '$(replace_cc_input_source)-V1-23E.npy')
    # contain the time series for each population.
    # We recommend using absolute paths rather than relative paths.
    'replace_cc_input_source': None,

    # whether to redistribute CC synapse to meet literature value
    # of E-specificity
    'E_specificity': True,

    # Relative inhibitory synaptic strength (in relative units).
    'g': -16.,

    # compute average indegree in V1 from data
    'av_indegree_V1': np.mean([av_indegree_Cragg, av_indegree_OKusky]),

    # synaptic volume density
    # area-specific --> conserves average in-degree
    # constant --> conserve syn. volume density
    'rho_syn': 'constant',

    # Increase the external Poisson indegree onto 5E and 6E
    'fac_nu_ext_5E': 1.,
    'fac_nu_ext_6E': 1.,
    # to increase the ext. input to 23E and 5E in area TH
    'fac_nu_ext_TH': 1.,

    # synapse weight parameters for current-based neurons
    # excitatory intracortical synaptic weight (mV)
    'PSP_e': 0.15,
    'PSP_e_23_4': 0.3,
    # synaptic weight (mV) for external input
    'PSP_ext': 0.15,

    # relative SD of normally distributed synaptic weights
    'PSC_rel_sd_normal': 0.1,
    # relative SD of lognormally distributed synaptic weights
    'PSC_rel_sd_lognormal': 3.0,

    # scaling factor for cortico-cortical connections (chi)
    'cc_weights_factor': 1.,
    # factor to scale cortico-cortical inh. weights in relation
    # to exc. weights (chi_I)
    'cc_weights_I_factor': 1.,

    # scaling factor for cortico-cortical connections (chi)
    'gba_weights_factor': 1,
    # factor to scale cortico-cortical inh. weights in relation to exc. weights (chi_I)
    'gba_weights_I_factor': 1,

    # Scale E to I
    'scalingEtoI': 1.,
    # Scale external weights to 5E
    'scaling5E': 1.,
    # Scale external weights to 6E
    'scaling6E': 1.,

    # Scale above threshold
    'eta_ext': 1.,

    # 'switch whether to distribute weights lognormally
    'lognormal_weights': False,
    # 'switch whether to distribute only EE weight lognormally if
    # 'lognormal_weights': True
    'lognormal_EE_only': False,

    # FB threshold for SLN
    'fb_threshold': .35,
    # FF threshold for SLN
    'ff_threshold': .65,
    # Whether cc connections are cluster specific
    'cluster_specific_cc_connections': False,
    # Whether cc are cluster
    'cluster_cc_connections': False,

    # mini clusters, big pool
    'mini_cluster': False,
    'neurons_per_minicolumn': 80,

    # Number of clusters for cc connections
    'Q': 1,
    # J_E_plus for cc connections
    'cc_J_E_plus': 1.,
    # R_j for cc connections
    'cc_R_J': 0.75,
}

network_params.update({'connection_params': connection_params})

"""
Analysis parameters
"""

ana_params = {}

ana_params['rate_histogram_binsize'] = 1.  # in ms
ana_params['extension'] = 'png'
ana_params['seed'] = 2106
ana_params['chu2014_path'] = None

ana_params['correlation_coefficient'] = {
    'subsample': 2000,
    'tbin': 1.,  # in ms
    'tmin': 500.,  # in ms
    'tmax': 1000.  # in ms
}

ana_params['compute_louvain'] = {
    'tmin': 500.  # in ms
}

ana_params['cv'] = {
    't_start': 500.,  # in ms
    't_stop': None  # in ms
}

ana_params['lvr'] = {
    't_start': 500.,  # in ms
    't_stop': None  # in ms
}

ana_params['plotAllFiringRatesSummary'] = {
    't_start': 500.,  # in ms
    't_stop': None,  # in ms
    'num_col': 4
    }

ana_params['plotAllBOLDSignalSummary'] = {
    'num_col': 4
    }

ana_params['plotConnectivities'] = {
    'tmin': 500.  # in ms
}

ana_params['plotBOLD'] = {
    'tmin': 500.,  # in ms
    'stepSize': 1.  # in ms
}

ana_params['plotRasterArea'] = {
    'fraction': 0.03,
    'low': 500,
    'high': 1000
}

network_params.update({'ana_params': ana_params})

"""
Stimulation parameters
"""
stim_params = {
    # Mean amplitude of the stimulating postsynaptic potential (in mV).
    'PSP_cluster': 0.15,
    # Mean amplitude of the stimulating postsynaptic potential (in mV).
    'PSP_stim': 0.15,
    # Standard deviation of the postsynaptic potential (in relative units).
    'PSP_sd': 0.1,
    # 3.3 % of al connections are from non-visual and subcortical structures,
    # 28,5 % of all connections are from within a patch. 3.3 + 28.5 = 31.8 of
    # all connections are not from within the model. Here we assume that all
    # non-visual/subcortical neurons act stimulation.The number of external
    # subcortical/non-visual connections should be roughly 3.3 / (28.5 + 3.3) *
    # K_ext
    'frac_type_4_vs_3': 3.3 / (28.5 + 3.3),
    # 3.3 % of al connections are from non-visual and subcortical structures,
    # 28,5 % of all connections are from within a patch. 3.3 + 28.5 = 31.8 of
    # all connections are not from within the model. Here we assume that all
    # non-visual/subcortical neurons act stimulation.The number of external
    # subcortical/non-visual connections should be roughly 3.3 / (28.5 + 3.3) *
    # K_ext
    'cluster_frac_type_4_vs_3': 3.3 / (28.5 + 3.3),
    }
stim_default_params = {
    # Start of the stimulating input (in ms).
    'stim_start': 600.0,
    # Duration of the stimulating input (in ms).
    'stim_duration': 100.0,
    # Rate of the stimulating input (in Hz).
    'stim_rate': 0.,
}
# Create stimulation dictionary for every area
stim_areas = {area: deepcopy(stim_default_params) for area in complete_area_list}
network_params.update({'stim_params': stim_params})
network_params.update({'stim_areas': stim_areas})

"""
Clustering
"""
cluster_default = {
        # =========================================
        # General clustering parameters for an area
        # =========================================
        'Q': 1,
        'J_E_plus': 1.,
        'R_J': 0.75,
        # =========================================
        # Generic stimulation input
        # =========================================
        'cluster_stimulation_parameters': {
            '0': {
                # Start of the stimulating input (in ms).
                'stim_start': 600.0,
                # Duration of the stimulating input (in ms).
                'stim_duration': 100.0,
                # Rate of the stimulating input (in Hz).
                'stim_rate': 0.,
                }
            },
        'pulvinar_stimulation_parameters': {
            '0': {
                # Start of the stimulating input (in ms).
                'stim_start': 600.0,
                # Duration of the stimulating input (in ms).
                'stim_duration': 100.0,
                # Rate of the stimulating input (in Hz).
                'stim_rate': 0.,
                }
            },
        # List of target layers of cluster specific simulation. Layers 4 and 6
        # are the thalamic inputs in the microcircuit
        'cluster_stim_target_population': deepcopy(population_list),
        # =========================================
        # Pulvinar input with specific target layer
        # =========================================
        # 3.3 % of al connections are from non-visual and subcortical structures,
        # 28,5 % of all connections are from within a patch. 3.3 + 28.5 = 31.8 of
        # all connections are not from within the model. Here we assume that all
        # non-visual/subcortical neurons stem from the pulvinar.
        # pulvinar connections should be roughly 3.3 / (28.5 + 3.3) * K_ext
        'pulvinar_frac_type_4_vs_3': 3.3 / (28.5 + 3.3),
        # There is evidence that pulvinar projects to synapses in deep layer 3
        'pulvinar_target_layers': ['23'],
        }
cluster_areas = {area: deepcopy(cluster_default) for area in complete_area_list}
cluster_areas.update({'external': deepcopy(cluster_default)})
cluster_areas.update({'stim': deepcopy(cluster_default)})
cluster_areas.update({'cluster_stim': deepcopy(cluster_default)})

network_params.update({'cluster': cluster_areas})

"""
Delays
"""
delay_params = {
    # Local dendritic delay for excitatory transmission [ms]
    'delay_e': 1.5,
    # Local dendritic delay for inhibitory transmission [ms]
    'delay_i': 0.75,
    # Relative standard deviation for both local and inter-area delays
    'delay_rel': 0.5,
    # Axonal transmission speed to compute interareal delays [mm/ms]
    'interarea_speed': 3.5
}
network_params.update({'delay_params': delay_params})

"""
Input parameters
"""
input_params = {
    # Whether to use Poisson or DC input (True or False)
    'poisson_input': True,

    # synapse type for Poisson input
    'syn_type_ext': 'static_synapse_hpc',

    # Rate of the Poissonian spike generator (in spikes/s).
    'rate_ext': 10.,

    'use_normal_rates': True,

    # Whether to switch on time-dependent DC input
    'dc_stimulus': False,
}

network_params.update({'input_params': input_params})

"""
Recording settings
"""
recording_dict = {
    # Which areas to record spike data from
    'areas_recorded': complete_area_list,

    # voltmeter
    'record_vm':  False,
    # Fraction of neurons to record membrane potentials from
    # in each population if record_vm is True
    'Nrec_vm_fraction': 0.01,

    # Fraction of neurons to record spikes from
    # in each population if reduced spike recording is requested
    'Nrec_spikes_fraction': 1.0,

    # Parameters for the spike detectors
    'spike_dict': {
        'label': 'spikes',
        'start': 0.},
    # Parameters for the voltmeters
    'vm_dict': {
        'label': 'vm',
        'start': 0.,
        'stop': 1000.,
        'interval': 0.1}
    }
if network_params['USING_NEST_3']:
    recording_dict['spike_dict'].update({'record_to': 'ascii'})
    recording_dict['vm_dict'].update({'record_to': 'ascii'})
else:
    recording_dict['spike_dict'].update({'withtime': True,
                                         'record_to': ['file']})
    recording_dict['vm_dict'].update({'withtime': True,
                                      'record_to': ['file']})
sim_params.update({'recording_dict': recording_dict})

"""
Theory params
"""

theory_params = {'neuron_params': neuron_params,
                 # Initial rates can be None (start integration at
                 # zero rates), a numpy.ndarray defining the initial
                 # rates or 'random_uniform' which leads to randomly
                 # drawn initial rates from a uniform distribution.
                 'initial_rates': None,
                 # If 'initial_rates' is set to 'random_uniform',
                 # 'initial_rates_iter' defines the number of
                 # different initial conditions
                 'initial_rates_iter': None,
                 # If 'initial_rates' is set to 'random_uniform',
                 # 'initial_rates_max' defines the maximum rate of the
                 # uniform distribution to draw the initial rates from
                 'initial_rates_max': 1000.,
                 # The simulation time of the mean-field theory integration
                 'T': 50.,
                 # The time step of the mean-field theory integration
                 'dt': 0.01,
                 'print_time': False,
                 # Weight for an additional external noise in mV. This can be
                 # used to drive the mean-field dynamics. Note that this needs
                 # a mdification of the nest source code to work, currently
                 # implemented in the siegert-inst_rate_conn branch of this
                 # repository: https://github.com/AlexVanMeegen/nest-simulator
                 'noise_weight': 0.,
                 # Time interval for recording the trajectory of the mean-field calcuation
                 # If None, then the interval is set to dt
                 'rec_interval': None}


"""
Helper function to update default parameters with custom
parameters
"""


def nested_update(d, d2):
    res = deepcopy(d)
    for key in d2:
        if isinstance(d2[key], dict) and key in res:
            res[key] = nested_update(res[key], d2[key])
        else:
            res[key] = deepcopy(d2[key])
    return res


def check_custom_params(d, def_d):
    for key, val in d.items():
        if isinstance(val, dict):
            check_custom_params(d[key], def_d[key])
        else:
            try:
                _ = def_d[key]
            except KeyError:
                raise KeyError('Unused key in custom parameter dictionary: {}'.format(key))
