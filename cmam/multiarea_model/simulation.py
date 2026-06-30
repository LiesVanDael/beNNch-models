
"""
multiarea_model
==============

Simulation class of the multi-area model of macaque visual vortex by
Schmidt et al. (2018).


Classes
-------
Simulation : Loads a parameter file that specifies simulation
parameters for a simulation of the instance of the model. A simulation
is identified by a unique hash label.

"""

import json
import nest
import numpy as np
import os
import pprint
import shutil
import time
import math
import pickle

from .analysis_helpers import _load_npy_to_dict, model_iter
from config import base_path, data_path
from copy import deepcopy
from .default_params import nested_update, sim_params
from .default_params import check_custom_params, network_params, complete_area_list 
from dicthash import dicthash
from .multiarea_helpers import extract_area_dict, create_vector_mask
try:
    from .sumatra_helpers import register_runtime
    sumatra_found = True
except ImportError:
    sumatra_found = False

class Simulation:
    def __init__(self, network, sim_spec, network_spec, data_folder_hash):
        """
        Simulation class.
        An instance of the simulation class with the given parameters.
        Can be created as a member class of a multiarea_model instance
        or standalone.

        Parameters
        ----------
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network to be simulated.
        params : dict
            custom simulation parameters that overwrite the
            default parameters defined in default_params.py
        """
        self.data_folder_hash = data_folder_hash
        self.params = deepcopy(sim_params)
        if isinstance(sim_spec, dict):
            self.custom_params = sim_spec
        else:
            fn = os.path.join(data_path,
                              sim_spec,
                              '_'.join(('custom_params',
                                        sim_spec)))

            # =================================================================
            # ================== Parallel reading in of data ==================
            # =================================================================
            with open(fn, 'r') as f:
                self.custom_params = json.load(f)['sim_params']

        self.params = nested_update(self.params, self.custom_params)
        
        self.network = network
        self.label = dicthash.generate_hash_from_dict({'params': self.params,
                                                       'network_label': self.network.label})
        print("Simulation label: {}".format(self.label))

        self.areas_simulated = self.params['areas_simulated']
        self.areas_recorded = self.params['recording_dict']['areas_recorded']
        self.T = self.params['t_sim']
        self.pre_T = self.params['t_presim']

        self.data_dir = os.path.join(data_path, self.data_folder_hash)
        if nest.Rank() == 0:
            try:
                os.makedirs(os.path.join(self.data_dir, 'recordings'))
            except OSError:
                pass
            self.copy_files()
            print("Copied files.")
            d = {'sim_params': self.custom_params,
                 'network_params': self.network.custom_params,
                 'network_label': self.network.label}
            with open(os.path.join(self.data_dir,
                                   'custom_params'), 'w') as f:
                json.dump(d, f)
            print("Initialized simulation class.")
            self.dump()

    def __eq__(self, other):
        # Two simulations are equal if the simulation parameters and
        # the simulated networks are equal.
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        s = "Simulation {} of network {} with parameters:".format(self.label, self.network.label)
        s += pprint.pformat(self.params, width=1)
        return s

    def dump(self):
        """Dump all data. If the first does not exist, then all others cannot exist."""
        fn = os.path.join(self.data_dir, 'W.pkl')
        if not os.path.isfile(fn):
            for name in ['W', 'W_sd', 'K', 'N', 'synapses', 'rates', 'J', 'add_DC_drive']:
                        obj = getattr(self.network, name)
                        if hasattr(obj, "to_dict"):
                            obj = obj.to_dict()
                        with open(os.path.join(self.data_dir, f'{name}.pkl'), 'wb') as f:
                            pickle.dump(obj, f)

    def copy_files(self):
        """
        Copy all relevant files for the simulation to its data directory.
        """
        files = [os.path.join('multiarea_model',
                              'data_multiarea',
                              'Model.py'),
                 os.path.join('multiarea_model',
                              'data_multiarea',
                              'VisualCortex_Data.py'),
                 os.path.join('multiarea_model',
                              'multiarea_model.py'),
                 os.path.join('multiarea_model',
                              'simulation.py'),
                 os.path.join('multiarea_model',
                              'default_params.py'),]
        if self.network.params['connection_params']['replace_cc_input_source'] is not None:
            fs = self.network.params['connection_params']['replace_cc_input_source']
            if '.json' in fs:
                files.append(fs)
            else:  # Assume that the cc input is stored in one npy file per population
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                for it in fn_iter:
                    fp_it = (fs,) + it
                    fp_ = '{}.npy'.format('-'.join(fp_it))
                    files.append(fp_)
        for f in files:
            shutil.copy2(os.path.join(base_path, f),
                         self.data_dir)

    def prepare(self):
        """
        Prepare NEST Kernel.
        """
        nest.ResetKernel()
        rng_seed = self.params['rng_seed']
        num_processes = int(self.params['num_processes'])
        local_num_threads = self.params['local_num_threads']
        vp = num_processes * local_num_threads
        nest.SetKernelStatus({'resolution': self.params['dt'],
                              'total_num_virtual_procs': vp,
                              'overwrite_files': True,
                              'data_path': os.path.join(self.data_dir, 'recordings'),
                              'print_time': False,                      
                              'rng_seed': rng_seed})
        nest.SetDefaults(self.network.params['neuron_params']['neuron_model'],
                         self.network.params['neuron_params']['single_neuron_dict'])
    
    def create_recording_devices(self):
        """
        Create devices for all populations. Depending on the
        configuration, this will create:
        - spike detector
        - voltmeter
        """
        try:
            self.spike_detector = nest.Create('spike_recorder')
        except:
            self.spike_detector = nest.Create('spike_detector')
        status_dict = deepcopy(self.params['recording_dict']['spike_dict'])
        label = '-'.join((self.label,
                          status_dict['label']))
        status_dict.update({'label': label})
        nest.SetStatus(self.spike_detector, status_dict)

        if self.params['recording_dict']['record_vm']:
            self.voltmeter = nest.Create('voltmeter')
            status_dict = self.params['recording_dict']['vm_dict']
            label = '-'.join((self.label,
                              status_dict['label']))
            status_dict.update({'label': label})
            nest.SetStatus(self.voltmeter, status_dict)

    def create_areas(self):
        """
        Create all areas with their populations and internal connections.
        """
        self.areas = []
        for area_name in self.areas_simulated:
            a = Area(self, self.network, area_name)
            self.areas.append(a)
            print("Memory after {0} : {1:.2f} MB".format(area_name, self.memory() / 1024.))

    def create_stimulation(self):
        """
        Create all stimulations.
        """
        for area in self.areas:
            area.connect_microstimulation()
            print("Memory after {0} : {1:.2f} MB".format(area.name, self.memory() / 1024.))

    def create_cluster_stimulation(self):
        """
        Create all stimulations.
        """
        for area in self.areas:
            area.connect_cluster_stimulation()
            print("Memory after {0} : {1:.2f} MB".format(area.name, self.memory() / 1024.))

    def create_pulvinar_stimulation(self):
        """
        Create all stimulations.
        """
        for area in self.areas:
            area.connect_pulvinar_stimulation()
            print("Memory after {0} : {1:.2f} MB".format(area.name, self.memory() / 1024.))

    def cortico_cortical_input(self):
        """
        Create connections between areas.
        """
        replace_cc = self.network.params['connection_params']['replace_cc']
        replace_non_simulated_areas = self.network.params['connection_params'][
            'replace_non_simulated_areas']
        if self.network.params['connection_params']['replace_cc_input_source'] is None:
            replace_cc_input_source = None
        else:
            replace_cc_input_source = os.path.join(self.data_dir,
                                                   self.network.params['connection_params'][
                                                       'replace_cc_input_source'])

        if not replace_cc and set(self.areas_simulated) != set(self.network.area_list):
            if replace_non_simulated_areas == 'het_current_nonstat':
                fn_iter = model_iter(mode='single', areas=self.network.area_list)
                non_simulated_cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
            elif replace_non_simulated_areas == 'het_poisson_stat':
                fn = self.network.params['connection_params']['replace_cc_input_source']
                with open(fn, 'r') as f:
                    non_simulated_cc_input = json.load(f)
            elif replace_non_simulated_areas == 'hom_poisson_stat':
                non_simulated_cc_input = {source_area_name:
                                          {source_pop:
                                           self.network.params['input_params']['rate_ext']
                                           for source_pop in
                                           self.network.structure[source_area_name]}
                                          for source_area_name in self.network.area_list}
            else:
                raise KeyError("Please define a valid method to"
                               " replace non-simulated areas.")

        if replace_cc == 'het_current_nonstat':
            fn_iter = model_iter(mode='single', areas=self.network.area_list)
            cc_input = _load_npy_to_dict(replace_cc_input_source, fn_iter)
        elif replace_cc == 'het_poisson_stat':
            with open(self.network.params['connection_params'][
                    'replace_cc_input_source'], 'r') as f:
                cc_input = json.load(f)
        elif replace_cc == 'hom_poisson_stat':
            cc_input = {source_area_name:
                        {source_pop:
                         self.network.params['input_params']['rate_ext']
                         for source_pop in
                         self.network.structure[source_area_name]}
                        for source_area_name in self.network.area_list}

        # Connections between simulated areas are not replaced
        if not replace_cc:
            for target_area in self.areas:
                # Loop source area though complete list of areas
                for source_area_name in self.network.area_list:
                    if target_area.name != source_area_name:
                        # If source_area is part of the simulated network,
                        # connect it to target_area
                        if source_area_name in self.areas:
                            source_area = self.areas[self.areas.index(source_area_name)]
                            connect(self,
                                    target_area,
                                    source_area)
                        # Else, replace the input from source_area with the
                        # chosen method
                        else:
                            target_area.create_additional_input(replace_non_simulated_areas,
                                                                source_area_name,
                                                                non_simulated_cc_input[
                                                                    source_area_name])
        # Connections between all simulated areas are replaced
        else:
            for target_area in self.areas:
                for source_area in self.areas:
                    if source_area != target_area:
                        target_area.create_additional_input(replace_cc,
                                                            source_area.name,
                                                            cc_input[source_area.name])

    def simulate(self):
        """
        Create the network and execute simulation.
        Record used memory and wallclock time.
        """

        t0 = time.time()
        self.base_memory = self.memory()
        self.prepare()
        t1 = time.time()
        self.time_kernel_prepare = t1 - t0
        print("Prepared simulation in {0:.2f} seconds.".format(self.time_kernel_prepare))

        self.create_recording_devices()
        self.create_areas()
        t2 = time.time()
        self.time_network_local = t2 - t1
        print("Created areas and internal connections in {0:.2f} seconds.".format(
            self.time_network_local))

        self.cortico_cortical_input()
        t3 = time.time()
        self.time_network_global = t3 - t2
        print("Created cortico-cortical connections in {0:.2f} seconds.".format(
            self.time_network_global))

        # Stimulates all clusters of an area
        self.create_stimulation()
        t4 = time.time()
        self.time_create_stimulation = t4 - t3
        print("Created stimulations in {0:.2f} seconds.".format(
            self.time_create_stimulation))

        # Stimulates only one cluster of an area
        self.create_cluster_stimulation()
        t5 = time.time()
        self.time_create_cluster_stimulation = t5 - t4
        print("Created cluster stimulations in {0:.2f} seconds.".format(
            self.time_create_cluster_stimulation))

        # Stimulates only one cluster of an area as pulvinar would
        self.create_pulvinar_stimulation()
        t6 = time.time()
        self.network_memory = self.memory()
        self.time_create_pulvinar_stimulation = t6 - t5
        print("Created pulvinar stimulations in {0:.2f} seconds.".format(
            self.time_create_pulvinar_stimulation))

        print(f'Calls to connect: {connect.call_counter}')
        print(f'Number of synapses: {connect.synapse_counter}')
        t7 = time.time()
        nest.Prepare()
        self.time_network_prepare = time.time() - t7
        print("Preparation took {0:.2f} seconds.".format(self.time_network_prepare))
        t8 = time.time()
        nest.Run(self.pre_T)
        self.time_presimulate = time.time() - t8
        print("Presimulated network in {0:.2f} seconds.".format(self.time_presimulate))
        self.intermediate_kernel_status = nest.kernel_status

        t8 = time.time()
        nest.Run(self.T)
        self.time_simulate = time.time() - t8
        self.total_memory = self.memory()
        print("Simulated network in {0:.2f} seconds.".format(self.time_simulate))
        self.logging()

    def memory(self):
        """
        Use NEST's memory wrapper function to record used memory.
        """
        return nest.memory_size 
    
    def logging(self):
        """
        Write runtime and memory for all MPI processes
        to file.
        """
        d = {'py_time_kernel_prepare': self.time_kernel_prepare,
             'py_time_network_local': self.time_network_local,
             'py_time_network_global': self.time_network_global,
             'py_time_create_stimulation': self.time_create_stimulation,
             'py_time_create_cluster_stimulation': self.time_create_cluster_stimulation,
             'py_time_create_pulvinar_stimulation': self.time_create_pulvinar_stimulation,
             'py_time_network_prepare': self.time_network_prepare,
             'py_time_presimulate': self.time_presimulate,
             'py_time_simulate': self.time_simulate,
             'base_memory': self.base_memory,
             'network_memory': self.network_memory,
             'total_memory': self.total_memory}

        final_kernel_status = nest.kernel_status
        d.update(final_kernel_status)
        print(f"final_kernel_status: {d}")

        # Subtract timer information from presimulation period
        presim_timers = ['time_collocate_spike_data', 'time_communicate_spike_data', 'time_deliver_secondary_data', 'time_deliver_spike_data', 'time_gather_secondary_data', 'time_gather_spike_data', 'time_omp_synchronization_simulation', 'time_mpi_synchronization', 'time_simulate', 'time_update']
        presim_timers.extend([timer + '_cpu' for timer in presim_timers])
        other_timers = ['time_communicate_prepare', 'time_communicate_target_data', 'time_construction_connect', 'time_construction_create', 'time_gather_target_data', 'time_omp_synchronization_construction']
        other_timers.extend([timer + '_cpu' for timer in other_timers])

        for timer in presim_timers:
            try:
                a = d[timer]
                pre = self.intermediate_kernel_status[timer]
                r = np.asarray(a) - np.asarray(pre)
                if isinstance(a, tuple):
                    d[timer] = tuple(r.tolist())
                else:
                    d[timer] = r.item() if r.ndim == 0 else tuple(r.tolist())
              #  d[timer + "_max"] = max(timer_array)
              #  d[timer + "_min"] = min(timer_array)
              #  d[timer + "_mean"] = np.mean(timer_array)
              #  d[timer + "_all"] = timer_array
              #  d[timer + '_presim'] = self.intermediate_kernel_status[timer]
              #  d[timer + "_presim_max"] = max(self.intermediate_kernel_status[timer])
              #  d[timer + "_presim_min"] = min(self.intermediate_kernel_status[timer])
              #  d[timer + "_presim_avg"] = np.mean(self.intermediate_kernel_status[timer])
              #  d[timer + "_presim_all"] = self.intermediate_kernel_status[timer]
            except KeyError:
                # KeyError if compiled without detailed timers, except time_simulate
                continue

        for timer in other_timers:
            try:
                a = d[timer]
                arr = np.asarray(a)
                if isinstance(a, tuple):
                    d[timer] = tuple(arr.tolist())
                else:
                    d[timer] = arr.item() if arr.ndim == 0 else tuple(arr.tolist())
            except KeyError:
                # KeyError if compiled without detailed timers, except time_simulate
                continue
        print(d)

        nest.Cleanup()


        fn = os.path.join(self.data_dir,
                          'recordings',
                          '_'.join((self.label,
                                    'logfile',
                                    str(nest.Rank()))))
        with open(fn, 'a') as f:
            for key, value in d.items():
                f.write(key + ' ' + str(value) + '\n')

    def save_network_gids(self): 
        with open(os.path.join(self.data_dir,
                               'recordings',
                               'network_gids.txt'), 'w') as f:
            for area in self.areas:
                for layer, d1 in self.network.N[area.name].items():
                    for pop, d2 in d1.items():
                        first_id = area.gids[(layer, pop)][0].get()['global_id']
                        last_id = area.gids[(layer, pop)][-1].get()['global_id']
                        f.write("{area},{layer},{pop},{g0},{g1}\n".format(area=area.name,
                                                                                        layer=layer,
                                                                                        pop=pop,
                                                                                        g0=first_id,
                                                                                        g1=last_id))

    def save_stim_gids(self):
        with open(os.path.join(self.data_dir,
                               'recordings',
                               'stim_gids.txt'), 'w') as f:
            for area in self.areas:
                for layer, d1 in self.network.N[area.name].items():
                    for pop, d2 in d1.items():
                        gid = area.cluster_stimulation_gid[(layer, pop)].get()['global_id']
                        f.write("{area},{layer},{pop},{g0}\n".format(area=area.name,
                                                                               layer=layer,
                                                                               pop=pop,
                                                                               g0=gid))

    def save_cluster_stim_gids(self):
        with open(os.path.join(self.data_dir,
                               'recordings',
                               'cluster_stimulation_gid.txt'), 'w') as f:
            for area in self.areas:
                for layer, d1 in self.network.N[area.name].items():
                    for pop, d2 in d1.items():
                        gid = area.pulvinar_gid[(layer, pop)].get()['global_id']
                        f.write("{area},{layer},{pop},{g0}\n".format(area=area.name,
                                                                               layer=layer,
                                                                               pop=pop,
                                                                               g0=gid))

    def save_pulvinar_gids(self):
        with open(os.path.join(self.data_dir,
                               'recordings',
                               'pulvinar_gid.txt'), 'w') as f:
            for area in self.areas:
                for layer, d1 in self.network.N[area.name].items():
                    for pop, d2 in d1.items():
                        for cluster in d2:
                            if self.network.params['USING_NEST_3']:
                                gid = area.pulvinar_gid[(layer, pop, cluster)].get()['global_id']
                            else:
                                gid = area.pulvinar_gid[(layer, pop, cluster)]
                            f.write("{area},{layer},{pop},{cluster},{g0}\n".format(area=area.name,
                                                                                   layer=layer,
                                                                                   pop=pop,
                                                                                   cluster=cluster,
                                                                                   g0=gid))

    def register_runtime(self):
        if sumatra_found:
            register_runtime(self.label)
        else:
            raise ImportWarning('Sumatra is not installed, the '
                                'runtime cannot be registered.')

class Area:
    def __init__(self, simulation, network, name):
        """
        Area class.
        This class encapsulates a single area of the model.
        It creates all populations and the intrinsic connections between them.
        It provides an interface to allow connecting the area to other areas.

        Parameters
        ----------
        simulation : simulation
           An instance of the simulation class that specifies the
           simulation that the area is part of.
        network : multiarea_model
            An instance of the multiarea_model class that specifies
            the network the area is part of.
        name : str
            Name of the area.
        """
        self.simulation = simulation
        self.network = network
        self.name = name
        self.stim_area_params = self.network.params['stim_areas'][self.name]
        # Contains information about pulvinar stimulation # TODO generalize naming
        self.cluster_stimulation_params = self.network.params['cluster'][self.name]['cluster_stimulation_parameters']
        self.pulvinar_params = self.network.params['cluster'][self.name]['pulvinar_stimulation_parameters']
        
        self.neuron_numbers = self.network.N[name]
        self.synapses = network.synapses[self.name]
        self.W = network.W[self.name]
        self.W_sd = network.W_sd[self.name]

        self.populations = []
        for t_layer, d1 in self.synapses.items():
            for t_pop, d2 in d1.items():
                self.populations.append((t_layer, t_pop)) 

        self.K_per_target_area = self.network.K[self.name]
        self.external_synapses = {t_layer: {t_pop: self.network.K[self.name][t_layer][t_pop]['external']['external']['external'] 
                                            for t_pop in self.network.K[self.name][t_layer]
                                            }
                                  for t_layer in self.network.K[self.name]
                                  }
         
        self.stim_synapses = {t_layer: {t_pop: self.network.K[self.name][t_layer][t_pop]['stim']['stim']['stim']
                                            for t_pop in self.network.K[self.name][t_layer]
                                            }
                                  for t_layer in self.network.K[self.name]
                                  }

        self.K_cluster_stim = {t_layer: {t_pop: self.network.K[self.name][t_layer][t_pop]['cluster_stim']['cluster_stim']['cluster_stim']
                                            for t_pop in self.network.K[self.name][t_layer]
                                         }
                                  for t_layer in self.network.K[self.name]
                                  }
        self.K_pulvinar = {t_layer: {t_pop: self.network.K[self.name][t_layer][t_pop]['pulvinar']['pulvinar']['pulvinar']
                                        for t_pop in self.network.K[self.name][t_layer]
                                            }
                                  for t_layer in self.network.K[self.name]
                                  }
        self.create_populations()
        self.connect_devices()
        self.connect_populations()
        print("Rank {}: created area {} with {} local nodes".format(nest.Rank(),
                                                                    self.name,
                                                                    self.num_local_nodes))

    def __str__(self):
        s = "Area {} with {} neurons.".format(
            self.name, int(self.neuron_numbers['total']))
        return s

    def __eq__(self, other):
        # If other is an instance of area, it should be the exact same
        # area This opens the possibility to have multiple instance of
        # one cortical areas
        if isinstance(other, Area):
            return self.name == other.name and self.gids == other.gids
        elif isinstance(other, str):
            return self.name == other

    def create_populations(self):
        """
        Create all populations of the area.
        """
        self.gids = {}
        self.num_local_nodes = 0
        for pop in self.populations:
            layer, population = pop
            total_neurons = self.neuron_numbers[layer][population]
            gid = nest.Create(self.network.params['neuron_params']['neuron_model'],
                              math.ceil(total_neurons))
            I_e = self.network.add_DC_drive[self.name][layer][population]
            if not self.network.params['input_params']['poisson_input']:
                # Sum external synapses across all clusters
                K_ext = self.external_synapses[layer][population]
                
                W_ext = self.network.W[self.name][layer][population]['external']['external']['external']
                tau_syn = self.network.params['neuron_params']['single_neuron_dict']['tau_syn_ex']
                DC = K_ext * W_ext * tau_syn * 1.e-3 * \
                    self.network.params['rate_ext']
                I_e += DC
            nest.SetStatus(gid, {'I_e': I_e})

            # Store GIDCollection of each population
            self.gids[(layer, population)] = gid
            # Initialize membrane potentials
            # This could also be done after creating all areas, which
            # might yield better performance. Has to be tested.
            gid.V_m = nest.random.normal(self.network.params['neuron_params']['V0_mean'],
                                         self.network.params['neuron_params']['V0_sd'])
   
    def connect_populations(self):
        """
        Create connections between populations.
        """
        connect(self.simulation,
                self,
                self)

    def connect_devices(self):
        if self.name in self.simulation.params['recording_dict']['areas_recorded']:
            # Check if only a fraction of spikes should be recorded
            spk_rec_fraction = self.simulation.params['recording_dict']['Nrec_spikes_fraction']
            reduced = spk_rec_fraction < 1.0
            # If so, determine the step size for selecting neurons
            if reduced:
                if spk_rec_fraction <= 0.0:
                    raise ValueError("record_fraction must be > 0 when reduced recording is requested")
                # use ceil so we do not record more than the requested fraction
                record_step = max(1, int(math.ceil(1.0 / spk_rec_fraction)))

            for pop in self.populations:
                layer, population = pop
                if reduced == False:
                    # Always record spikes from all neurons to get correct
                    # statistics
                    nest.Connect(self.gids[(layer, population)],
                                 self.simulation.spike_detector)
                else:
                    # Only record from a fraction of the neurons to reduce data volume
                    nest.Connect(self.gids[(layer, population)][::record_step],
                                 self.simulation.spike_detector)

        if self.simulation.params['recording_dict']['record_vm']:
            for pop in self.populations:
                layer, population = pop
                total_neurons = self.neuron_numbers[layer][population]
                nrec = int(self.simulation.params['recording_dict']['Nrec_vm_fraction'] * total_neurons)
                nest.Connect(self.simulation.voltmeter,
                             self.gids[(layer, population)][:nrec])

        if self.network.params['input_params']['poisson_input']:
            self.poisson_generators = []
            for pop in self.populations:
                layer, population = pop
                K_ext = self.external_synapses[layer][population] 
                W_ext = self.network.W[self.name][layer][population]['external']['external']['external']
                pg = nest.Create('poisson_generator')
                nest.SetStatus(
                    pg, {'rate': self.network.rates[self.name][layer][population] * K_ext})
                syn_spec = {'weight': W_ext}
                nest.Connect(pg,
                             self.gids[(layer, population)],
                             syn_spec=syn_spec)
                self.poisson_generators.append(pg[0])

    def connect_microstimulation(self):
        """ Connects the microstimulation population to the corresponding areas."""
        if nest.Rank() == 0:
            print('Microstimulation connection established')
        for pop in self.populations:
            layer, population = pop
            K_stim = self.stim_synapses[layer][population]
            W_stim = self.network.W[self.name][layer][population]['stim']['stim']['stim']
            W_stim_sd = self.network.W_sd[self.name][layer][population]['stim']['stim']['stim']
            self.stop_stim = (self.stim_area_params['stim_start'] +
                              self.stim_area_params['stim_duration'])

            poisson_stim = nest.Create('poisson_generator')
            self.microstimulation_gid = poisson_stim[0]
            nest.SetStatus(
                poisson_stim, {
                    'rate': self.stim_area_params['stim_rate'] * K_stim,
                    'start': self.stim_area_params['stim_start'],
                    'stop': self.stop_stim
                    }
                )

            if self.stim_area_params['stim_rate'] > 0.:
                print('area', self.name)
                print('layer', pop[0])
                print('population', pop[1])
                print('rate', self.stim_area_params['stim_rate'])
            
            syn_spec = {
                    'synapse_model': 'static_synapse',
                    'weight': nest.random.normal(
                        mean=W_stim,
                        std=W_stim_sd
                        )
                      }

            nest.Connect(poisson_stim,
                         self.gids[(layer, population)],
                         syn_spec=syn_spec
                         )

    def connect_cluster_stimulation(self):
        """ Connects the microstimulation population to the corresponding areas."""
        if nest.Rank() == 0:
            print('Cluster stimulation connection established')
        self.cluster_stimulation_gid = {}
        for pop in self.populations:
            # pop contains layer, population, cluster
            layer, population = pop
            K = self.K_cluster_stim[layer][population]
            W = self.network.W[self.name][layer][population]['cluster_stim']['cluster_stim']['cluster_stim']
            W_sd = self.network.W_sd[self.name][layer][population]['cluster_stim']['cluster_stim']['cluster_stim']

            stim_start = self.cluster_stimulation_params['0']['stim_start']
            stim_rate = self.cluster_stimulation_params['0']['stim_rate']
            stim_duration = self.cluster_stimulation_params['0']['stim_duration']

            if isinstance(stim_rate, list):
                # stim_rate and stim_start have to be lists of the same length
                # stim_duration is not needed in this case
                # Example:
                # rate_times = [10., 20., 30.] and rate_values = [12., 0., 16.]
                # means that until t=10 rate is zero, between t=10 and t=20 the
                # rate is 12, between t=20 and t=30 the rate is 0, and from
                # t=30 on the rate is 16
                rate = [r * K for r in stim_rate]

                poisson_stim = nest.Create('inhomogeneous_poisson_generator')
                self.cluster_stimulation_gid[(layer, population)] = poisson_stim[0]
                nest.SetStatus(
                    poisson_stim, {
                        'rate_times': stim_start,
                        'rate_values': rate,
                        }
                    )
                if len(stim_rate) > 0:
                    print('area', self.name)
                    print('layer', layer)
                    print('population', population)
                    print('rate', rate)
                    print('stim_start', stim_start)
            else:
                stim_stop = (stim_start + stim_duration)

                rate = stim_rate * K

                poisson_stim = nest.Create('poisson_generator')
                self.cluster_stimulation_gid[(layer, population)] = poisson_stim[0]
                nest.SetStatus(
                    poisson_stim, {
                        'rate': rate,
                        'start': stim_start,
                        'stop': stim_stop
                        }
                    )

                if stim_rate > 0.:
                    print('area', self.name)
                    print('layer', layer)
                    print('population', population)
                    print('rate', stim_rate)
            
            syn_spec = {
                    'synapse_model': 'static_synapse',
                    'weight': nest.random.normal(
                        mean=W,
                        std=W_sd
                        )
                      }

            nest.Connect(poisson_stim,
                         self.gids[(layer, population)],
                         syn_spec=syn_spec
                         )

    def connect_pulvinar_stimulation(self):
        """ Connects the pulvinar population to the corresponding areas."""
        if nest.Rank() == 0:
            print('Cluster stimulation connection established')
        self.pulvinar_gid = {}
        for pop in self.populations:
            # pop contains layer, population, cluster
            layer, population = pop
            K = self.K_pulvinar[layer][population]
            W = self.network.W[self.name][layer][population]['pulvinar']['pulvinar']['pulvinar']
            W_sd = self.network.W_sd[self.name][layer][population]['pulvinar']['pulvinar']['pulvinar']

            stim_start = self.pulvinar_params['0']['stim_start']
            stim_rate = self.pulvinar_params['0']['stim_rate']
            stim_duration = self.pulvinar_params['0']['stim_duration']
            if isinstance(stim_rate, list):
                # stim_rate and stim_start have to be lists of the same length
                # stim_duration is not needed in this case
                # Example:
                # rate_times = [10., 20., 30.] and rate_values = [12., 0., 16.]
                # means that until t=10 rate is zero, between t=10 and t=20 the
                # rate is 12, between t=20 and t=30 the rate is 0, and from
                # t=30 on the rate is 16
                rate = [r * K for r in stim_rate]

                poisson_stim = nest.Create('inhomogeneous_poisson_generator')
                self.pulvinar_gid[(layer, population)] = poisson_stim[0]
                nest.SetStatus(
                    poisson_stim, {
                        'rate_times': stim_start,
                        'rate_values': rate,
                        }
                    )
                if len(stim_rate) > 0:
                    print('area', self.name)
                    print('layer', layer)
                    print('population', population)
                    print('rate', rate)
                    print('stim_start', stim_start)
            else:
                stim_stop = (stim_start + stim_duration)

                rate = stim_rate * K

                poisson_stim = nest.Create('poisson_generator')
                self.pulvinar_gid[(layer, population)] = poisson_stim[0]
                nest.SetStatus(
                    poisson_stim, {
                        'rate': rate,
                        'start': stim_start,
                        'stop': stim_stop
                        }
                    )

                if stim_rate > 0.:
                    print('area', self.name)
                    print('layer', layer)
                    print('population', population)
                    print('rate', stim_rate)
                    print('stim_start', stim_start)
            
            syn_spec = {
                      'synapse_model': 'static_synapse',
                      'weight': nest.random.normal(
                          mean=W,
                          std=W_sd
                          )
                        }

            nest.Connect(poisson_stim,
                         self.gids[(layer, population)],
                         syn_spec=syn_spec
                         )

    def create_additional_input(self, input_type, source_area_name, cc_input):
        """
        Replace the input from a source area by the chosen type of input.

        Parameters
        ----------
        input_type : str, {'het_current_nonstat', 'hom_poisson_stat',
                           'het_poisson_stat'}
            Type of input to replace source area. The source area can
            be replaced by Poisson sources with the same global rate
            rate_ext ('hom_poisson_stat') or by specific rates
            ('het_poisson_stat') or by time-varying specific current
            ('het_current_nonstat')
        source_area_name: str
            Name of the source area to be replaced.
        cc_input : dict
            Dictionary of cortico-cortical input of the process
            replacing the source area.
        """
        synapses = extract_area_dict(self.network.synapses, #TODO: why is this function used here
                                     self.network.structure,
                                     self.name,
                                     source_area_name)
        W = extract_area_dict(self.network.W,
                              self.network.structure,
                              self.name,
                              source_area_name)

        v = self.network.params['delay_params']['interarea_speed']
        s = self.network.distances[self.name][source_area_name]
        delay = s / v
        for pop in self.populations:
            layer, population = pop
            for source_pop in self.network.structure[source_area_name]:
                target_key = f"{layer}{population}"
                syn_spec = {'weight': W[target_key][source_pop],
                            'delay': delay}
                # Sum K across all clusters and normalize by total neurons
                total_neurons = self.neuron_numbers[layer][population]
                K = synapses[target_key][source_pop] / total_neurons
                if input_type == 'het_current_nonstat':
                    curr_gen = nest.Create('step_current_generator')
                    dt = self.simulation.params['dt']
                    T = self.simulation.params['t_sim']
                    assert(len(cc_input[source_pop]) == int(T))
                    nest.SetStatus(curr_gen, {'amplitude_values': K * cc_input[source_pop] * 1e-3,
                                              'amplitude_times': np.arange(dt,
                                                                           T + dt,
                                                                           1.)})
                    nest.Connect(curr_gen,
                                 self.gids[(layer, population)],
                                 syn_spec=syn_spec)
                elif 'poisson_stat' in input_type:  # hom. and het. poisson lead here
                    pg = nest.Create('poisson_generator')
                    nest.SetStatus(pg, {'rate': K * cc_input[source_pop]})
                    nest.Connect(pg,
                                 self.gids[(layer, population)],
                                 syn_spec=syn_spec)

def connect(simulation,
            target_area,
            source_area,
            ):
    """
    Connect two areas with each other.

    Parameters
    ----------
    simulation : Simulation instance
        Simulation simulating the network containing the two areas.
    target_area : Area instance
        Target area of the projection
    source_area : Area instance
        Source area of the projection
    """
    network = simulation.network

    cluster_cc_connections = network.params['connection_params']['cluster_cc_connections'] # cluster connections across areas
    
    for (layer_t, pop_t) in target_area.populations:
        for (layer_s, pop_s) in source_area.populations:

            # Number of synapses
            number_of_synapses = math.ceil(
                network.synapses[target_area.name][layer_t][pop_t][source_area.name][layer_s][pop_s]
                )

            if number_of_synapses > 0:
                conn_spec = {'rule': 'alt_clustered_fixed_total_number',
                             'N': number_of_synapses,
                             'num_clusters': network.params['connection_params']['Q']}
                
                if target_area == source_area:
                    J_E_plus = network.params['cluster'][target_area.name]['J_E_plus']
                    J_E_minus = network.params['cluster'][target_area.name]['J_E_minus']
                    J_I_plus = network.params['cluster'][target_area.name]['J_I_plus'] 
                    J_I_minus = network.params['cluster'][target_area.name]['J_I_minus']
                    if pop_s == 'E':
                        w_min = 0.
                        w_max = np.inf
                        mean_delay = network.params['delay_params']['delay_e']
                    elif pop_s == 'I':
                        w_min = -np.inf
                        w_max = 0.
                        mean_delay = network.params['delay_params']['delay_i']
                else:
                    w_min = 0.
                    w_max = np.inf
                    v = network.params['delay_params']['interarea_speed']
                    s = network.distances[target_area.name][source_area.name]
                    mean_delay = s / v
                    if cluster_cc_connections:
                        J_E_plus = network.params['connection_params']['cc_J_E_plus']
                        J_E_minus = network.params['connection_params']['cc_J_E_minus']
                        J_I_plus = network.params['connection_params']['cc_J_I_plus']
                        J_I_minus = network.params['connection_params']['cc_J_I_minus']
                    else:
                        J_E_plus = 1.
                        J_E_minus = 1.
                        J_I_plus = 1.
                        J_I_minus = 1.

                # Mean synaptic weight
                mean_weight = network.W[target_area.name][layer_t][pop_t][source_area.name][layer_s][pop_s]
                if pop_t == 'E' and pop_s == 'E':
                    mean_weight_plus = mean_weight*J_E_plus
                    mean_weight_minus = mean_weight*J_E_minus
                else:
                    mean_weight_plus = mean_weight*J_I_plus
                    mean_weight_minus = mean_weight*J_I_minus

                
                # Standard deviation of weight
                std_weight = network.W_sd[target_area.name][layer_t][pop_t][source_area.name][layer_s][pop_s]

                syn_spec = {
                    'synapse_model': 'static_synapse',
                    'delay': nest.math.redraw(
                        nest.random.normal(
                            mean=mean_delay,
                            std=mean_delay * network.params['delay_params']['delay_rel']
                            ),
                        min=simulation.params['dt'],
                        max=np.inf)}
               
                connect.call_counter += 1
                connect.synapse_counter += number_of_synapses

                nest.Connect(source_area.gids[(layer_s, pop_s)],
                             target_area.gids[(layer_t, pop_t)],
                             conn_spec,
                             nest.CollocatedSynapses({**syn_spec, 
                                                      'weight': nest.math.redraw(
                                                          nest.random.normal(
                                                          mean=mean_weight_plus,
                                                          std=std_weight
                                                          ),
                                                      min=w_min,
                                                      max=w_max
                                                      )},
                             {**syn_spec, 'weight': nest.math.redraw(
                                 nest.random.normal(
                                     mean=mean_weight_minus,
                                     std=std_weight
                                     ),
                                 min=w_min,
                                 max=w_max)}),
                             )

connect.call_counter = 0
connect.synapse_counter = 0

