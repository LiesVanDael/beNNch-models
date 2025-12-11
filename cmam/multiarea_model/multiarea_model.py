"""
multiarea_model
==============

Network class to instantiate and administer instances of the
multi-area model of macaque visual cortex by Schmidt et al. (2018).

Classes
-------
MultiAreaModel : Loads a parameter file that specifies custom parameters for a
particular instance of the model. An instance of the model has a unique hash
label. As members, it may contain three classes:

- simulation : contains all relevant parameters for a simulation of
  the network

- theory : theory class that serves to estimate the stationary state
  of the network using mean-field theory

  Schuecker J, Schmidt M, van Albada SJ, Diesmann M, Helias M (2017)
  Fundamental Activity Constraints Lead to Specific Interpretations of
  the Connectome. PLoS Comput Biol 13(2): e1005179.
  doi:10.1371/journal.pcbi.1005179

- analysis: provides methods to load data and perform data analysis

"""
import json
import numpy as np
import os
import pprint
import shutil
import pandas as pd
import time
import nest
import pickle
from .default_params import complete_area_list, nested_update, network_params
from .default_params import check_custom_params, sim_params
from collections import OrderedDict
from copy import deepcopy
from .data_multiarea.Model import compute_Model_params
from .analysis import Analysis
from config import base_path, data_path
from dicthash import dicthash
from nested_dict import nested_dict
from .multiarea_helpers import (
    load_degree_data_nested_dict,
    filter_nested_dict_3,
    filter_nested_dict_6,
    calculate_K_synapses_nested_dict,
)
from .simulation import Simulation
from .theory import Theory


# Set precision of dicthash library to 1e-4
# because this is sufficient for indegrees
# and neuron numbers and guarantees reproducibility
# of the class label despite inevitably imprecise float calculations
# in the data scripts.
dicthash.FLOAT_FACTOR = 1e4
dicthash.FLOOR_SMALL_FLOATS = True


class MultiAreaModel:
    def __init__(self, network_spec, theory=False, simulation=False,
                 analysis=False, *args, **keywords):
        """
        Multiarea model class.
        An instance of the multiarea model with the given parameters.

        Parameters
        ----------
        network_spec : dict or str
            Specify the network. If it is of type dict, the parameters defined
            in the dictionary overwrite the default parameters defined in
            default_params.py.
            If it is of type str, the string defines the label of a previously
            initialized model instance that is now loaded.
        theory : bool
            whether to create an instance of the theory class as member.
        simulation : bool
            whether to create an instance of the simulation class as member.
        analysis : bool
            whether to create an instance of the analysis class as member.

        """
        self.params = deepcopy(network_params)

        # =======================================================
        # Initialize network from dictionary or reopen from label
        # =======================================================
        if isinstance(network_spec, dict):
            print("Initializing network from dictionary.")
            # check_custom_params(network_spec, self.params)
            self.custom_params = network_spec
            p_ = 'multiarea_model/data_multiarea/custom_data_files'
            # Draw random integer label for data script to avoid clashes with
            # parallelly created class instances
            rand_data_label = np.random.randint(10000)
            print("RAND_DATA_LABEL", rand_data_label)
            tmp_parameter_fn = os.path.join(base_path,
                                            p_,
                                            'custom_{}_parameter_dict.json'.format(rand_data_label))
            
            tmp_data_fn = os.path.join(base_path,
                                       p_,
                                       'custom_Data_Model_{}.json'.format(rand_data_label))
            
            tmp_structure_fn = os.path.join(base_path,
                                       p_,
                                       'custom_structure_{}.json'.format(rand_data_label))

            tmp_distances_fn = os.path.join(base_path,
                                       p_,
                                       'custom_distances_{}.json'.format(rand_data_label))

            tmp_area_list_fn = os.path.join(base_path,
                                       p_,
                                       'custom_area_list_{}.json'.format(rand_data_label))

            with open(tmp_parameter_fn, 'w') as f:
                json.dump(self.custom_params, f)
            self.params = nested_update(self.params, self.custom_params)
            self.params = self.add_clusters_to_params(self.params) 
            # Execute Data script
            ts = time.time()
            compute_Model_params(
                    out_label=str(rand_data_label),
                    mode='custom',
                    CLUSTER=self.params['cluster']
                    )
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'compute_Model_params took {passed_time} s')

            with open(tmp_data_fn, 'r') as f:
                dat = json.load(f)

            # ===============================================
            # Extract and set parameters from the loaded data
            # ===============================================

            self.structure = OrderedDict()
            for area in dat['area_list']:
                self.structure[area] = dat['structure'][area]

            self.N = dat['neuron_numbers'] # 3 layers deep
            self.synapses = dat['synapses'] # 6 layers deep
            self.W = dat['synapse_weights_mean'] # 6 layers deep
            self.W_sd = dat['synapse_weights_sd'] # 6 layers deep
            self.area_list = complete_area_list
            self.distances = dat['distances']
            self.cluster_params = dat['cluster_params'] # LVD 

            # ===========================================================
            # Filter out thalamic populations and corresponding connections
            # ===========================================================            
            ts = time.time()
            
            self.N = filter_nested_dict_3(self.N, remove_TH_layer_4=True)
            self.synapses = filter_nested_dict_6(self.synapses, remove_TH_layer_4=True)
            self.W = filter_nested_dict_6(self.W, remove_TH_layer_4=True)
            self.W_sd = filter_nested_dict_6(self.W_sd, remove_TH_layer_4=True)
            
            ind, _, _, _ = load_degree_data_nested_dict(self.N, self.synapses) # LVD

            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'transformation of nested dicts took {passed_time} s') 
            
            # =========================
            # Rescale E to I, 5E and 6E
            # =========================
            # TODO: can be simplified
            ts = time.time()

            scalingEtoI = self.params['connection_params']['scalingEtoI']

            for a_t, d1 in self.W.items():
                for layer_t, d2 in d1.items():
                    if 'I' in d2:
                        for a_s, d3 in d2['I'].items():
                            for layer_s, d4 in d3.items():
                                if 'E' in d4:
                                    original_val = d4['E']
                                    d4['E'] = original_val * scalingEtoI

            for a_t, d1 in self.W_sd.items():
                for layer_t, d2 in d1.items():
                    if 'I' in d2:
                        for a_s, d3 in d2['I'].items():
                            for layer_s, d4 in d3.items():
                                if 'E' in d4:
                                    original_val = d4['E']
                                    d4['E'] = original_val * scalingEtoI

            scaling5E = self.params['connection_params']['scaling5E']

            for a_t in self.W:
                for layer_t in self.W[a_t]:
                    if layer_t == '5':
                        for pop_t in self.W[a_t][layer_t]:
                            if pop_t == 'E':
                                for a_s in self.W[a_t][layer_t][pop_t]:
                                    if a_s == 'external':
                                        for layer_s in self.W[a_t][layer_t][pop_t][a_s]:
                                            if layer_s == 'external':
                                                for pop_s in self.W[a_t][layer_t][pop_t][a_s][layer_s]:
                                                    if pop_s == 'external':
                                                        self.W[a_t][layer_t][pop_t][a_s][layer_s][pop_s] *= scaling5E

            scaling6E = self.params['connection_params']['scaling6E']

            for a_t in self.W:
                for layer_t in self.W[a_t]:
                    if layer_t == '6':
                        for pop_t in self.W[a_t][layer_t]:
                            if pop_t == 'E':
                                for a_s in self.W[a_t][layer_t][pop_t]:
                                    if a_s == 'external':
                                        for layer_s in self.W[a_t][layer_t][pop_t][a_s]:
                                            if layer_s == 'external':
                                                for pop_s in self.W[a_t][layer_t][pop_t][a_s][layer_s]:
                                                    if pop_s == 'external':
                                                        self.W[a_t][layer_t][pop_t][a_s][layer_s][pop_s] *= scaling6E # LVD 

            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'scaling 5E 6E EtoI took {passed_time} s')

            # ==================================================================
            # Rescale cc-connections (chi) and cc-connections inh vs exc (chi_I)
            # ==================================================================
            # TODO: can be simplified
            ts = time.time()

            # scaling factor for cortico-cortical connections (chi)
            gba_weights_factor = self.params['connection_params']['gba_weights_factor']
            # factor to scale cortico-cortical inh. weights in relation to exc. weights (chi_I)
            gba_weights_I_factor = self.params['connection_params']['gba_weights_I_factor']
            for source in self.area_list:
                for target in self.area_list:
                    if source != target:
                        if not np.isclose(gba_weights_factor, 1.):
                            # CC connections
                            for a_t, d1 in self.W.items():
                                if a_t != target:
                                    continue
                                for layer_t, d2 in d1.items():
                                    for pop_t, d3 in d2.items():
                                        if pop_t != 'E':
                                            continue
                                        for a_s, d4 in d3.items():
                                            if a_s != source:
                                                continue
                                            for layer_s, d5 in d4.items():
                                                for pop_s, value in d5.items():
                                                    if pop_s != 'E':
                                                        continue
                                                    self.W[a_t][layer_t][pop_t][a_s][layer_s][pop_s] *= gba_weights_factor 
                                                    self.W_sd[a_t][layer_t][pop_t][a_s][layer_s][pop_s] *= gba_weights_factor 
                    else:
                        if not np.isclose(gba_weights_I_factor, 1.):
                            # inter areal connections
                            for a_t, d1 in self.W.items():
                                if a_t != target:
                                    continue
                                for layer_t, d2 in d1.items():
                                    for pop_t, d3 in d2.items():
                                        if pop_t != 'E':
                                            continue
                                        for a_s, d4 in d3.items():
                                            if a_s != source:
                                                continue
                                            for layer_s, d5 in d4.items():
                                                for pop_s, value in d5.items():
                                                    if pop_s != 'I':
                                                        continue
                                                    self.W[a_t][layer_t][pop_t][a_s][layer_s][pop_s] *= gba_weights_I_factor 
                                                    self.W_sd[a_t][layer_t][pop_t][a_s][layer_s][pop_s] *= gba_weights_I_factor 

            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'gba took {passed_time} s')
            # ==================================================================
            # If K_stable is specified in the params, load the stabilized matrix
            # ==================================================================
            ts = time.time()

            # TODO: Extend this by calling the stabilization method
            if self.params['connection_params']['K_stable'] is None:
                self.K = ind
            else:
                if not isinstance(self.params['connection_params']['K_stable'], str):
                    raise TypeError("Not supported. Please store the "
                                    "matrix in a binary numpy file and define "
                                    "the path to the file as the parameter value.")

                # Assume that the parameter defines a filename containing the matrix
                K_stable = np.load(self.params['connection_params']['K_stable'])
                self.K, self.synapses = calculate_K_synapses_nested_dict( # LVD 
                        self.synapses,
                        self.area_list,
                        self.structure,
                        self.params['connection_params']['K_stable'],
                        self.N,
                        ind,
                        )

            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'calculation of K and synapses took {passed_time} s')
            # ===================================
            # Set background rates
            # The base case uses the same input rate for every area. By setting
            # use_normal_rates to false one can set a specific rate for every
            # area. This feature has not been extensively tested.
            # ===================================
            ts = time.time()

            use_normal_rates = self.params['input_params']['use_normal_rates']

            self.rates = nested_dict()
            
            if use_normal_rates:
                rate_ext = self.params['input_params']['rate_ext']
                for a in self.K:
                    for l in self.K[a]:
                        for p in self.K[a][l]:
                            self.rates[a][l][p] = rate_ext
            else:                
                tau_m = self.params['neuron_params']['single_neuron_dict']['tau_m']
                tau_syn = self.params['neuron_params']['single_neuron_dict']['tau_syn_ex']
                C_m = self.params['neuron_params']['single_neuron_dict']['C_m']
                V_th = self.params['neuron_params']['single_neuron_dict']['V_th']
                E_L = self.params['neuron_params']['single_neuron_dict']['E_L']
                eta_ext = self.params['connection_params']['eta_ext']
               # self.rates = \
               #     (1e3 / (tau_m * K_ext * tau_syn * W_ext / C_m))*(V_th - E_L) * eta_ext
                for a in self.K:
                    for l in self.K[a]:
                       for p in self.K[a][l]:
                           try:
                               K_ext = self.K[a][l][p]['external']['external']['external']
                               W_ext = self.W[a][l][p]['external']['external']['external']
                               val = ( 
                                       1e3 / (tau_m * K_ext * tau_syn * W_ext / C_m)) * (V_th - E_L) * eta_ext
                               self.rates[a][l][p] = val
                           except KeyError:
                                print("Missing external input for {a}-{l}-{p}, setting rate to 0.")
                                self.rates[a][l][p] = 0.0

            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'setting of rates took {passed_time} s')

            # ===========================================================
            # Optionally scale down and adjust all parameters accordingly
            # ===========================================================
            ts = time.time()


            # Calculate J, needed for DC input calculation
            tau_syn_ex = self.params['neuron_params']['single_neuron_dict']['tau_syn_ex']
            C_m = self.params['neuron_params']['single_neuron_dict']['C_m']

            # LVD 
            self.J = nested_dict()
            for t_area, d1 in self.W.items():
                for t_layer, d2 in d1.items():
                    for t_pop, d3 in d2.items():
                        for s_area, d4 in d3.items():
                            for s_layer, d5 in d4.items():
                                for s_pop, val in d5.items():
                                    self.J[t_area][t_layer][t_pop][s_area][s_layer][s_pop] = (
                                        val * tau_syn_ex / C_m
                                    )


            # --- intitialize DC drive
            self.add_DC_drive = nested_dict()
            for a, d1 in self.N.items():
                for l, d2 in d1.items():
                    for p, _ in d2.items():
                        self.add_DC_drive[a][l][p] = 0.0

            # --- scaling
            if self.params['K_scaling'] != 1 or self.params['N_scaling'] != 1:
                if isinstance(self.params['fullscale_rates'], np.ndarray):
                    raise ValueError("Not supported. Please store the "
                                     "rates in a file and define the path to the file as "
                                     "the parameter value.")
                else:
                    with open(self.params['fullscale_rates'], 'r') as f:
                        full_mean_rates_dict_tmp = json.load(f)
                    full_mean_rates_dict = nested_dict()
                    for area, d in full_mean_rates_dict_tmp.items():
                        for pop, val in d.items():
                            if pop == 'total':
                                full_mean_rates_dict[area][pop][pop] = val
                            else:
                                layer = pop[:-1]
                                p = pop[-1]
                                full_mean_rates_dict[area][layer][p] = val

                fmr_tmp = filter_nested_dict_3(full_mean_rates_dict, remove_TH_layer_4=True, remove_total=True)

                tau_m = self.params['neuron_params']['single_neuron_dict']['tau_m']
                rate_ext = self.rates
                K_scaling = self.params['K_scaling']

                # Scale the number of neurons
                for a, d1 in self.N.items():
                    for l, d2 in d1.items():
                        for p, val in d2.items():
                            self.N[a][l][p] = val * self.params['N_scaling']

                # Scale the number of synapses
                for t_area, d1 in self.synapses.items():
                    for t_layer, d2 in d1.items():
                        for t_pop, d3 in d2.items():
                            for s_area, d4 in d3.items():
                                for s_layer, d5 in d4.items():
                                    for s_pop, val in d5.items():
                                        self.synapses[t_area][t_layer][t_pop][s_area][s_layer][s_pop] = (
                                            val * self.params['K_scaling'] * self.params['N_scaling']
                                        )

                # Calculate x1_ext_df and x1_df which are needed for scaling the DC drive
                
                # test code below
                for t_area, d1 in self.K.items():
                    for t_layer, d2 in d1.items():
                        for t_pop, d3 in d2.items():
                            # --- external input ---
                            try:
                                Kext = self.K[t_area][t_layer][t_pop]['external']['external']['external']
                                Jext = self.J[t_area][t_layer][t_pop]['external']['external']['external']
                                rate_ext_val = self.rates[t_area][t_layer][t_pop]
                                x1_ext_val = 1e-3 * tau_m * rate_ext_val * Kext * Jext
                            except KeyError:
                                x1_ext_val = 0.0

                            # --- recurrent input ---
                            x1_val = 0.0
                            for s_area, d4 in d3.items():
                                if s_area == 'external':
                                    continue
                                for s_layer, d5 in d4.items():
                                    for s_pop, K_val in d5.items():
                                        J_val = self.J[t_area][t_layer][t_pop][s_area][s_layer][s_pop]
                                        #fmr = fmr_tmp[s_area][s_layer][s_pop]
                                        fmr = fmr_tmp.get(s_area, {}).get(s_layer, {}).get(s_pop, 0.0) 
                                        x1_val += 1e-3 * tau_m * K_val * J_val * fmr

                            # --- final drive for this target population ---
                            #add_DC_drive.setdefault(t_area, {}).setdefault(t_layer, {})[t_pop] = (
                            correction = C_m / tau_m * (1.0 - np.sqrt(K_scaling)) * (x1_ext_val + x1_val)
                            self.add_DC_drive[t_area][t_layer][t_pop] = correction
                            #)

                # Scale J
                # self.J /= np.sqrt(K_scaling) # LVD
                for a_t, d1 in self.J.items():
                    for layer_t, d2 in d1.items():
                        for pop_t, d3 in d2.items():
                            for a_s, d4 in d3.items():
                                for layer_s, d5 in d4.items():
                                    for pop_s, val in d5.items():
                                        self.J[a_t][layer_t][pop_t][a_s][layer_s][pop_s] /= np.sqrt(K_scaling)
                                        self.W[a_t][layer_t][pop_t][a_s][layer_s][pop_s] = self.J[a_t][layer_t][pop_t][a_s][layer_s][pop_s] * C_m / tau_syn_ex
                # Scale K
                for a_t, d1 in self.K.items():
                    for layer_t, d2 in d1.items():
                        for pop_t, d3 in d2.items():
                            for a_s, d4 in d3.items():
                                for layer_s, d5 in d4.items():
                                    for pop_s in d5:
                                        self.K[a_t][layer_t][pop_t][a_s][layer_s][pop_s] *= K_scaling

            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'downscaling took {passed_time} s')
            # LVD 
            # Initialize result
            self.K_areas = nested_dict() 

            for t_area in self.K:
                if t_area not in self.K_areas:
                    self.K_areas[t_area] = {}
                # Sum over target layers and populations
                total_N_t = sum(self.N[t_area][t_layer][t_pop] 
                                for t_layer in self.N[t_area] 
                                for t_pop in self.N[t_area][t_layer])
                
                for s_area in {val for t_layer in self.K[t_area].values() 
                                   for t_pop in t_layer.values() 
                                   for val in t_pop}:
                    # Sum weighted by N
                    weighted_sum = 0
                    for t_layer in self.K[t_area]:
                        for t_pop in self.K[t_area][t_layer]:
                            for s_layer in self.K[t_area][t_layer][t_pop][s_area]:
                                for s_pop in self.K[t_area][t_layer][t_pop][s_area][s_layer]:
                                    weighted_sum += self.K[t_area][t_layer][t_pop][s_area][s_layer][s_pop] \
                                                    * self.N[t_area][t_layer][t_pop]
                    # Normalize by total target area population
                    self.K_areas[t_area][s_area] = weighted_sum / total_N_t

            def dict_to_sorted_tuple(d):
                """Recursively convert nested dict to sorted tuples for consistent hashing."""
                if isinstance(d, dict):
                    return tuple((k, dict_to_sorted_tuple(d[k])) for k in sorted(d))
                return d  # base case: value

            # Prepare hashable dictionary
            hash_dict = {
                'params': self.params,
                'K': dict_to_sorted_tuple(self.K),
                'N': dict_to_sorted_tuple(self.N),
                'structure': self.structure
            }

            # Generate hash, ignoring blacklisted keys
            self.label = dicthash.generate_hash_from_dict(
                hash_dict,
                blacklist=[
                    ('params', 'fullscale_rates'),
                    ('params', 'connection_params', 'K_stable'),
                    ('params', 'ana_params'),
                    ('params', 'connection_params', 'replace_cc_input_source')
                ]
            )

            # ================
            # Create filenames
            # ================

            if isinstance(network_spec, dict):
                parameter_fn = os.path.join(base_path,
                                            'config_files',
                                            '{}_config'.format(self.label))
                data_fn = os.path.join(base_path,
                                       'config_files',
                                       'custom_Data_Model_{}.json'.format(self.label))
                structure_fn = os.path.join(base_path,
                                       'config_files',
                                       'custom_structure_{}.json'.format(self.label))
                distances_fn = os.path.join(base_path,
                                       'config_files',
                                       'custom_distances_{}.json'.format(self.label))
                area_list_fn = os.path.join(base_path,
                                       'config_files',
                                       'custom_area_list_{}.json'.format(self.label))

                shutil.move(tmp_parameter_fn,
                            parameter_fn)
                shutil.move(tmp_data_fn,
                            data_fn)
                shutil.move(tmp_structure_fn,
                            structure_fn)
                shutil.move(tmp_distances_fn,
                            distances_fn)
                shutil.move(tmp_area_list_fn,
                            area_list_fn)
            elif isinstance(network_spec, str):
                assert(network_spec == self.label)
            CLUSTER = self.params['cluster']

            CLUSTER['pulvinar'] = deepcopy(CLUSTER['cluster_stim'])
            
            self.W = clusterize_4(self.W, CLUSTER)
            self.K = clusterize_4(self.K, CLUSTER, K=True) # TODO
            self.N = clusterize_2(self.N, CLUSTER) # TODO: CLUSTER
            self.synapses = clusterize_4(self.synapses, CLUSTER, syn=True)
            self.J = clusterize_4(self.J, CLUSTER)
            
            self.W = apply_clustering_strengths(self.W, CLUSTER, self.params) # TODO: check whether right params
            self.J = apply_clustering_strengths(self.J, CLUSTER, self.params)

        else:
            print("Initializing network from label.")
            parameter_fn = os.path.join(base_path,
                                        'config_files',
                                        '{}_config'.format(network_spec))
            tmp_data_fn = os.path.join(base_path,
                                       'config_files',
                                       'custom_Data_Model_{}.json'.format(network_spec))
            structure_fn = os.path.join(base_path,
                                   'config_files',
                                   'custom_structure_{}.json'.format(network_spec))
            distances_fn = os.path.join(base_path,
                                   'config_files',
                                   'custom_distances_{}.json'.format(network_spec))
            area_list_fn = os.path.join(base_path,
                                   'config_files',
                                   'custom_area_list_{}.json'.format(network_spec))

            if 'sim_spec' not in keywords:
                sim_spec = {}
            else:
                sim_spec = keywords['sim_spec']

            sim_label = dicthash.generate_hash_from_dict(
                    {'params': nested_update(sim_params, sim_spec),
                     'network_label': network_spec
                     })
            sim_path = os.path.join(data_path, sim_label)

            self.label = network_spec

            # =================================================================
            # ================== Parallel reading in of data ==================
            # =================================================================

            with open(parameter_fn, 'r') as f:
                self.custom_params = json.load(f) # TODO: check: is this the same custom_params as in initialization simulation?
            with open(area_list_fn, 'r') as f:
                area_list = json.load(f)
            with open(structure_fn, 'r') as f:
                structure = json.load(f)
                self.structure = OrderedDict()
                for area in area_list:
                    self.structure[area] = structure[area]
            with open(distances_fn, 'r') as f:
                self.distances = json.load(f)
            # ===============================================
            # Extract and set parameters from the loaded data
            # ===============================================
            ts = time.time()

            self.area_list = complete_area_list

            self.params = nested_update(self.params, self.custom_params)
            self.params = self.add_clusters_to_params(self.params)

            self.load_dumps(sim_path, self.params)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'loading dumps took {passed_time} s')
        


        # =========================
        # Initialize member classes
        # =========================

        ts = time.time()
        if theory:
            if 'theory_spec' not in keywords:
                theory_spec = {}
            else:
                theory_spec = keywords['theory_spec']
            self.init_theory(theory_spec)
        te = time.time()
        passed_time = round(te - ts, 3)
        print(f'init theory took {passed_time} s')

        ts = time.time()
        if simulation:
            if 'sim_spec' not in keywords:
                sim_spec = {}
            else:
                sim_spec = keywords['sim_spec']
            self.init_simulation(sim_spec, network_spec) # LVD
        te = time.time()
        passed_time = round(te - ts, 3)
        print(f'init simulation took {passed_time} s')

        ts = time.time()
        if analysis:
            assert(getattr(self, 'simulation'))
            if 'ana_spec' not in keywords:
                ana_spec = {}
            else:
                ana_spec = keywords['ana_spec']
            self.init_analysis(ana_spec)
        te = time.time()
        passed_time = round(te - ts, 3)
        print(f'init analysis took {passed_time} s')

    def __str__(self):
        s = "Multi-area network {} with custom parameters: \n".format(self.label)
        s += pprint.pformat(self.params, width=1)
        return s

    def __eq__(self, other):
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def init_theory(self, theory_spec):
        self.theory = Theory(self, theory_spec)

    def init_simulation(self, sim_spec, network_spec):
        self.simulation = Simulation(self, sim_spec, network_spec) # LVD 

    def init_analysis(self, ana_spec):
        assert(hasattr(self, 'simulation'))
        if 'load_areas' in ana_spec:
            load_areas = ana_spec['load_areas']
        else:
            load_areas = None
        if 'data_list' in ana_spec:
            data_list = ana_spec['data_list']
        else:
            data_list = ['spikes']
        self.analysis = Analysis(self, self.simulation,
                                 data_list=data_list,
                                 load_areas=load_areas)

    def add_clusters_to_params(self, params):
        for area, p in params['cluster'].items():
            Q = p['Q']
            CLUSTER = [str(q) for q in range(Q)]
            J_E_minus = J_I_plus = J_I_minus = 1
            J_E_plus = p['J_E_plus']
            R_J = p['R_J']
            assert J_E_plus <= Q
            if Q != 1:

                J_E_minus = (Q - J_E_plus) / (Q - 1)
                J_I_plus = 1 + R_J * (J_E_plus - 1)
                J_I_minus = (Q - J_I_plus) / (Q - 1)

            params = nested_update(
                    params,
                    {'cluster': {area: {
                        'CLUSTER': CLUSTER,
                        'J_E_minus': J_E_minus,
                        'J_I_plus': J_I_plus,
                        'J_I_minus': J_I_minus,
                        }}}
                    )
        cluster_cc_connections = \
            self.params['connection_params']['cluster_cc_connections']
        cc_Q = params['connection_params']['Q']
        cc_J_E_minus = cc_J_I_plus = cc_J_I_minus = 1
        if cluster_cc_connections and cc_Q > 1:
            cc_J_E_plus = params['connection_params']['cc_J_E_plus']
            cc_R_J = params['connection_params']['cc_R_J']
            cc_J_E_minus = (cc_Q - cc_J_E_plus) / (cc_Q - 1)
            cc_J_I_plus = 1 + cc_R_J * (cc_J_E_plus - 1)
            cc_J_I_minus = (cc_Q - cc_J_I_plus) / (cc_Q - 1)
        params = nested_update(
                params,
                {'connection_params': {
                    'cc_J_E_minus': cc_J_E_minus,
                    'cc_J_I_plus': cc_J_I_plus,
                    'cc_J_I_minus': cc_J_I_minus,
                    }}
                )
        return params

    
    def make_cc_connections_directed(self, d): # TODO 
        tmp_dict = nested_dict()
        for t_area, d1 in d.items():
            tmp = 0
            for source, val in d0.items():
                (t_area, t_layer, t_population, t_cluster) = target
                (s_area, s_layer, s_population, s_cluster) = source
                last_cluster = str(self.params['cluster'][t_area]['Q'] - 1)

                if s_area in complete_area_list:
                    if s_area is t_area:
                        tmp_dict[(t_area, t_layer, t_population, t_cluster)][(s_area, s_layer, s_population, s_cluster)] = val
                    else:
                        tmp += val
                        if s_cluster == last_cluster:
                            tmp_dict[(t_area, t_layer, t_population, t_cluster)][(s_area, s_layer, s_population, t_cluster)] = tmp
                            tmp = 0
                else:
                    tmp_dict[(t_area, t_layer, t_population, t_cluster)][(s_area, s_layer, s_population, s_cluster)] = val

        result = pd.DataFrame(tmp_dict).fillna(0.)
        result.index.names = ['area', 'layer', 'population', 'cluster']
        result.columns.names = ['area', 'layer', 'population', 'cluster']
        return result
    
    def make_cc_connections_directed_nested_dict(self, d): # TODO 
        tmp_dict = nested_dict()
        for t_area, d1 in d.items():
            tmp = 0
            for source, val in d0.items():
                (t_area, t_layer, t_population, t_cluster) = target
                (s_area, s_layer, s_population, s_cluster) = source
                last_cluster = str(self.params['cluster'][t_area]['Q'] - 1)

                if s_area in complete_area_list:
                    if s_area is t_area:
                        tmp_dict[(t_area, t_layer, t_population, t_cluster)][(s_area, s_layer, s_population, s_cluster)] = val
                    else:
                        tmp += val
                        if s_cluster == last_cluster:
                            tmp_dict[(t_area, t_layer, t_population, t_cluster)][(s_area, s_layer, s_population, t_cluster)] = tmp
                            tmp = 0
                else:
                    tmp_dict[(t_area, t_layer, t_population, t_cluster)][(s_area, s_layer, s_population, s_cluster)] = val

        result = tmp_dict
        return result

    def _load_pickled_attrs(self, sim_path, names):
        """Load a set of pickled attributed into self."""
        for name in names:
            with open(os.path.join(sim_path, f"{name}.pkl"), "rb") as f:
                obj = pickle.load(f)
            # If it's a plain dict, wrap it back into nested_dict # LVD
            if isinstance(obj, dict) and not isinstance(obj, nested_dict):
                obj = nested_dict(obj)
            setattr(self, name, obj)

    def load_dumps(self, sim_path, params):
        """
        Load all data dumps rank safe

        Letting all ranks read in the same data can lead to ranks accessing the
        same file at the same time which makes the reading in fail. This
        routine lets ranks sequentially read in data. First rank 0 reads in
        data and touches a file as soon as it finishes. This is seen by rank 1
        which starts to read in data and touches a file as soon as it finished.
        This is done until all ranks read in their data.
        """
        CLUSTER = params['cluster']
        
        attrs = ['W', 'W_sd', 'K', 'N', 'synapses', 'rates', 'J', 'add_DC_drive']

        self._load_pickled_attrs(sim_path, attrs)

        # =========================================================================
        # Optionally connect cluster i in area A exclusively to cluster i in area B
        # =========================================================================
        ts = time.time()

        cluster_specific_cc_connections = \
            self.params['connection_params']['cluster_specific_cc_connections']

       # if cluster_specific_cc_connections:
       #     self.K = self.make_cc_connections_directed_nested_dict(self.K) # LVD TODO: what is K, how to change function 
       #     self.synapses = self.make_cc_connections_directed_nested_dict(self.synapses) # LVD moved here

        te = time.time()
        passed_time = round(te - ts, 3)
        print(f'cluster specific cc connections took {passed_time} s')

        # =========================================================================
        # Mini Clusters with big pool
        # =========================================================================
        ts = time.time()


        mini_cluster = \
            self.params['connection_params']['mini_cluster']
        if mini_cluster:
            assert(self.params['connection_params']['K_stable'] is not None)

        te = time.time()
        passed_time = round(te - ts, 3)
        print(f'mini clusters took {passed_time} s')
    
    def get_scaling_df(
            self,
            J_E_plus,
            J_E_minus,
            J_I_plus,
            J_I_minus,
            Q
            ):

        idx = pd.IndexSlice
        w_scale = pd.DataFrame(
                index=self.W.loc['V1', 'V1'].index,
                columns=self.W.loc['V1', 'V1'].columns
                )
        for x in range(Q):
            for y in range(Q):
                if x == y:
                    w_scale.loc[
                            idx[:, 'E', str(x)],
                            idx[:, 'E', str(x)]
                            ] = J_E_plus
                    w_scale.loc[
                            idx[:, 'E', str(x)],
                            idx[:, 'I', str(x)]
                            ] = J_I_plus
                    w_scale.loc[
                            idx[:, 'I', str(x)],
                            idx[:, 'E', str(x)]
                            ] = J_I_plus
                    w_scale.loc[
                            idx[:, 'I', str(x)],
                            idx[:, 'I', str(x)]
                            ] = J_I_plus
                else:
                    w_scale.loc[
                            idx[:, 'E', str(x)],
                            idx[:, 'E', str(y)]
                            ] = J_E_minus
                    w_scale.loc[
                            idx[:, 'E', str(x)],
                            idx[:, 'I', str(y)]
                            ] = J_I_minus
                    w_scale.loc[
                            idx[:, 'I', str(x)],
                            idx[:, 'E', str(y)]
                            ] = J_I_minus
                    w_scale.loc[
                            idx[:, 'I', str(x)],
                            idx[:, 'I', str(y)]
                            ] = J_I_minus
        if self.params['connection_params']['mini_cluster']:
            # Connections that stay inside the background cluster (0) remain
            # unclustered.
            w_scale.loc[idx[:, :, '0'], :] = 1.
            w_scale.loc[:, idx[:, :, '0']] = 1.
        return w_scale

    def mini_cluster_big_pool_N4(self, K_stable_path, ind): # N 4 levels (with clusters) # TODO
        neurons_per_minicolumn = self.params['connection_params']['neurons_per_minicolumn']
        n_per_cluster = nested_dict()
        N_ = nested_dict()
        for a, d1 in self.N.items():
            for layer, d2 in d1.items():
                for pop, d3 in d2.items():
                    for cluster, val in d3.items():
                        n_per_cluster[a][cluster] += val
        q_ = self.params['cluster']['V1']['Q']
        tmp = nested_dict()
        for a, d1 in n_per_cluster.items():
            for cluster, n in d1.items():
                tmp[a][cluster] = neurons_per_minicolumn / n / q_

        N_ = nested_dict()

        for a, d1 in self.N.items():
            for layer, d2 in d1.items():
                for pop, d3 in d2.items():
                   v = np.full(q_, tmp[a]['0'])
                   v[0] = 1 - sum(v[1:])
                   v *= q_
                   for i, cluster in enumerate(d3.keys()):
                       N_[a][layer][pop][cluster] = self.N[area][layer][pop][cluster]*v[i]

       # for key_, n_ in self.N.groupby(['area', 'layer', 'population']):
       #     v = np.full(q_, tmp[(key_[0], '0')])
       #     v[0]  = 1 - sum(v[1:])
       #     v *= q_
       #     N_[key_] = n_*v

        K_stable = np.load(K_stable_path)
        x = nested_dict()
        source_areas = set()
        for a_t, d1 in self.synapses.items():
            for layer_t, d2 in d1.items():
                for pop_t, d3 in d2.items():
                    for a_s in d3:
                        source_areas.add(a_s)
        external_input = list(
                source_areas - set(self.area_list)
                )
        i = 0
        # target area
        for a1 in self.area_list:
            # target layer and target population
            for s1 in self.structure[a1]:
                # target cluster
                # for c1 in self.params['cluster'][a1]['CLUSTER']: # remove clusters
                layer1 = s1[:-1]
                pop1 = s1[-1]
                j = 0
                # source area
                for a2 in self.area_list:
                    # source layer and source population
                    for s2 in self.structure[a2]:
                        # source cluster
                        # for c2 in self.params['cluster'][a2]['CLUSTER']: # remove clusters
                        layer2 = s2[:-1]
                        pop2 = s2[-1]

                        sum_K = 0.0
                        for _ in range(q_):
                            sum_K += K_stable[i][j] / q_
                            j += 1
                        #divider = N_[a2, layer2, pop2].sum() / N_[a2, layer2, pop2, c2]
                        #divider = sum(N_[a2][layer2][pop2].values()) # correct??? 
                        x[a1][layer1][pop1][a2][layer2][pop2] = sum_K
                        #j += 1
                for inp in external_input:
                    x[a1][layer1][pop1][inp][inp][inp] = ind[a1][layer1][pop1][inp][inp][inp]
                i += 1
        synapses = nested_dict()
        for a_t, d1 in x.items():
            for layer_t, d2 in d1.items():
                for pop_t, d3 in d2.items():
                    n_val = sum(N_[a_t][layer_t][pop_t].values())
                    for a_s, d4 in d3.items():
                            for pop_s, val in d5.items():
                                synapses[a_t][layer_t][pop_t][a_s][layer_s][pop_s] = val*n_val

        return self.N, synapses, x 


def clusterize_4(d, cluster_params, syn=False, K=False):
    new_dict = nested_dict()
    for a_t, d1 in d.items():
        cluster_t = cluster_params[a_t]['CLUSTER']
        for layer_t, d2 in d1.items():
            for pop_t, d3 in d2.items():
                for c0 in cluster_t:
                    for a_s, d4 in d3.items():
                        cluster_s = cluster_params[a_s]['CLUSTER']
                        divider = 1.
                        if syn:
                            divider = 1. / cluster_params[a_t]['Q'] / cluster_params[a_s]['Q']
                        if K:
                            divider = 1./ cluster_params[a_s]['Q']
                        for layer_s, d5 in d4.items():
                            for pop_s, d6 in d5.items():
                                val = d6 * divider
                                if pop_s == 'E' or pop_s == 'I':
                                    for c1 in cluster_s:
                                        new_dict[a_t][layer_t][pop_t][c0][a_s][layer_s][pop_s][c1] = val
                                else:
                                    new_dict[a_t][layer_t][pop_t][c0][a_s][layer_s][pop_s][pop_s] = val
    return new_dict

def clusterize_2(d, cluster_params): 
    new_dict = nested_dict()
    for a, d1 in d.items():
        cluster = cluster_params[a]['CLUSTER'] 
        num_cluster = cluster_params[a]['Q'] 
        for layer, d2 in d1.items():
            for pop, d3 in d2.items():
                for c0 in cluster:
                    new_dict[a][layer][pop][c0] = d3 / num_cluster
    return new_dict

def apply_clustering_strengths(W, CLUSTER, params):
    # =============================================
    # Apply clustering strengths JE+, JI+, JE-, JI-
    # =============================================
    ts = time.time()
    cluster_cc_connections = \
            params['connection_params']['cluster_cc_connections'] # cluster connections across areas
    
    for t_area in complete_area_list:
        for t_layer in W[t_area]:
            for t_pop in W[t_area][t_layer]:
                for t_c in W[t_area][t_layer][t_pop]:
                    for s_area in complete_area_list: 
                        if (s_area == t_area):
                            J_E_plus = params['cluster']['V1']['J_E_plus']
                            J_E_minus = params['cluster']['V1']['J_E_minus']
                            J_I_plus = params['cluster']['V1']['J_I_plus'] 
                            J_I_minus = params['cluster']['V1']['J_I_minus']
                        elif cluster_cc_connections:
                            J_E_plus = params['connection_params']['cc_J_E_plus']
                            J_E_minus = params['connection_params']['cc_J_E_minus']
                            J_I_plus = params['connection_params']['cc_J_I_plus']
                            J_I_minus = params['connection_params']['cc_J_I_minus']
                        else: #TODO check
                            continue
                        for s_layer in W[t_area][t_layer][t_pop][t_c][s_area]:
                            for s_pop in W[t_area][t_layer][t_pop][t_c][s_area][s_layer]:
                                for s_c, val in W[t_area][t_layer][t_pop][t_c][s_area][s_layer][s_pop].items():
                                    if t_c == s_c:
                                        if t_pop == 'E' and s_pop == 'E':
                                            W[t_area][t_layer][t_pop][t_c][s_area][s_layer][s_pop][s_c] *= J_E_plus
                                        else:
                                            W[t_area][t_layer][t_pop][t_c][s_area][s_layer][s_pop][s_c] *= J_I_plus
                                    else:
                                        if t_pop == 'E' and s_pop == 'E':
                                            W[t_area][t_layer][t_pop][t_c][s_area][s_layer][s_pop][s_c] *= J_E_minus
                                        else:
                                            W[t_area][t_layer][t_pop][t_c][s_area][s_layer][s_pop][s_c] *= J_I_minus
    
    te = time.time()
    passed_time = round(te - ts, 3)
    print(f'setting of clustered weights (cluster_cc_connections is {cluster_cc_connections}) took {passed_time} s')
    return W
