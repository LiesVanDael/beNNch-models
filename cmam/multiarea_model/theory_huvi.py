# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import yaml
from dicthash import dicthash

import nest
from nest.lib.hl_api_exceptions import NESTError


class Theory():
    """
    Calculates firing rates using mean field theory (Fourcaud & Brunel 2002).
    See also Schuecker et al. (2017).

    Parameters
    ----------
    theo_dict : dict
        dictionary containing all parameters specific to the theory
        such as the number of timesteps for the auxilliary ode.
        (see: default_theo_params.py)
    net_dict : dict
         dictionary containing all parameters specific to the neurons
         and the network (see: network_params.py)
    """

    def __init__(self, M):
        self.theo_dict = M.theory.params
        self.net_dict = M.params
        self.M = M

        # get parameters for siegert neurons
        lif_params = self.net_dict['neuron_params']['single_neuron_dict']
        assert lif_params['tau_syn_ex'] == lif_params['tau_syn_in']
        self.neuron_params = {
            'tau_m': lif_params['tau_m'],
            'tau_syn': lif_params['tau_syn_ex'],
            't_ref': lif_params['t_ref'],
            'theta': lif_params['V_th'] - lif_params['E_L'],
            'V_reset': lif_params['V_reset'] - lif_params['E_L']
        }

    def setup_nest(self, num_threads):
        """
        Hands parameters to the NEST-kernel.

        Resets the NEST-kernel and passes parameters to it.
        The number of seeds for the NEST-kernel is computed, based on the
        total number of MPI processes and threads of each.

        Parameters
        ----------
        num_threads : int
            Local number of threads (per MPI process).
        """
        nest.ResetKernel()
        nest.SetKernelStatus({'local_num_threads': num_threads})
        N_tp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        self.sim_resolution = self.theo_dict['dt']
        kernel_dict = {
            'resolution': self.sim_resolution,
            'rng_seed': self.net_dict['sim_params']['rng_seed'],
            'print_time': self.theo_dict['print_time'],
            'use_wfr': False
            }
        nest.SetKernelStatus(kernel_dict)

    def create_populations(self):
        """
        Creates the neuronal populations.

        A siegert neuron for each population is created
        and the parameters are assigned to them.
        """
        # Create cortical populations.
        self.pops = {}
        for pop, nn in self.M.N.iteritems():
            if nn > 0:
                population = nest.Create('siegert_neuron', 1)
                nest.SetStatus(population, self.neuron_params)
                self.pops[pop] = population

    def create_devices(self):
        """
        Creates the recording device.
        """
        interval = self.theo_dict['rec_interval']
        if interval is None:
            interval = self.theo_dict['dt']

        self.multimeter = nest.Create('multimeter', params={
            'record_from': ['rate'],
            'interval': interval
        })

    def create_poisson(self):
        """
        Creates siegert neurons corresponding to the Poisson generators.
        """
        self.poissons = {}
        for pop in self.pops:
            poiss = nest.Create('siegert_neuron', 1)
            nest.SetStatus(poiss, self.neuron_params)
            rate_ext = self.M.rates.loc[pop]
            nest.SetStatus(poiss, {'rate': rate_ext, 'mean': rate_ext})
            self.poissons[pop] = poiss

    def create_noise(self):
        """
        Creates additional unit-strength OU noise for every population to
        drive the dynamics.
        """
        self.noises = {}
        noise_params = {'tau': self.neuron_params['tau_syn'], 'lambda': 1.,
                        'mu': 0., 'sigma': 1.}
        for pop in self.pops:
            noise = nest.Create('lin_rate_ipn', 1)
            nest.SetStatus(noise, noise_params)
            self.noises[pop] = noise

    def connect_neurons(self):
        """
        Connects the neuronal populations.
        """
        tau_m = self.net_dict['neuron_params']['single_neuron_dict']['tau_m']
        tau_syn = self.net_dict['neuron_params']['single_neuron_dict']['tau_syn_ex']
        C_m = self.net_dict['neuron_params']['single_neuron_dict']['C_m']
        for (area_i, layer_i, pop_i, cluster_i), target_pop in self.pops.items():
            for (area_j, layer_j, pop_j, cluster_j), source_pop in self.pops.items():
                synapse_nr = self.M.synapses.loc[
                    (area_j, layer_j, pop_j, cluster_j),
                    (area_i, layer_i, pop_i, cluster_i)
                ]
                target_nn = self.M.N.loc[
                    (area_i, layer_i, pop_i, cluster_i)
                ]

                if synapse_nr > 0 and target_nn > 0:
                    indeg = synapse_nr / target_nn
                    weight = self.M.W.loc[
                        (area_j, layer_j, pop_j, cluster_j),
                        (area_i, layer_i, pop_i, cluster_i)
                    ]
                    weight *= tau_syn / C_m
                    # 1e-3 to make [tau] = ms consistent with [nu] = 1/s
                    syn_dict = {
                        'drift_factor': 1e-3*tau_m*indeg*weight,
                        'diffusion_factor': 1e-3*tau_m*indeg*weight**2,
                        'synapse_model': 'diffusion_connection',
                        'receptor_type': 0
                    }
                    try:
                        nest.Connect(
                            source_pop, target_pop,
                            conn_spec='all_to_all',
                            syn_spec=syn_dict
                        )
                    except NESTError:
                        syn_dict['model'] = syn_dict.pop('synapse_model')
                        nest.Connect(
                            source_pop, target_pop,
                            conn_spec='all_to_all',
                            syn_spec=syn_dict
                        )

    def connect_poisson(self):
        """
        Connects the Poisson generators to the populations.
        """
        tau_m = self.net_dict['neuron_params']['single_neuron_dict']['tau_m']
        tau_syn = self.net_dict['neuron_params']['single_neuron_dict']['tau_syn_ex']
        C_m = self.net_dict['neuron_params']['single_neuron_dict']['C_m']
        for (area_i, layer_i, pop_i, cluster_i), target_pop in self.pops.items():
            synapse_nr = self.M.synapses.loc[
                ('external', 'external', 'external', 'external'),
                (area_i, layer_i, pop_i, cluster_i)
            ]
            target_nn = self.M.N.loc[
                (area_i, layer_i, pop_i, cluster_i)
            ]

            if synapse_nr > 0 and target_nn > 0:
                indeg = synapse_nr / target_nn
                weight = self.M.W.loc[
                ('external', 'external', 'external', 'external'),
                (area_i, layer_i, pop_i, cluster_i)
                ]
                weight *= tau_syn / C_m
                # 1e-3 to make [tau] = ms consistent with [nu] = 1/s
                syn_dict_pois = {
                    'drift_factor': 1e-3*tau_m*indeg*weight,
                    'diffusion_factor': 1e-3*tau_m*indeg*weight**2,
                    'synapse_model': 'diffusion_connection',
                    'receptor_type': 0
                }
                pois = self.poissons[(area_i, layer_i, pop_i, cluster_i)]
                try:
                    nest.Connect(
                        pois, target_pop,
                        conn_spec='all_to_all',
                        syn_spec=syn_dict_pois
                    )
                except NESTError:
                    syn_dict_pois['model'] = syn_dict_pois.pop('synapse_model')
                    nest.Connect(
                        pois, target_pop,
                        conn_spec='all_to_all',
                        syn_spec=syn_dict_pois
                    )

    def connect_noise(self):
        """
        Connects the noise to the respective populations.
        """
        for (area_i, layer_i, pop_i, cluster_i), target_pop in self.pops.items():
            syn_dict_noise = {
                'weight': self.theo_dict['noise_weight'], 'receptor_type': 0,
                'synapse_model': 'rate_connection_instantaneous'
            }
            try:
                nest.Connect(
                    self.noises[(area_i, layer_i, pop_i, cluster_i)], target_pop,
                    conn_spec='one_to_one', syn_spec=syn_dict_noise
                )
            except NESTError:
                syn_dict_noise['model'] = syn_dict_noise.pop('synapse_model')
                nest.Connect(
                    self.noises[(area_i, layer_i, pop_i, cluster_i)], target_pop,
                    conn_spec='one_to_one', syn_spec=syn_dict_noise
                )

    def connect_devices(self):
        """ Connects the recording device to the microcircuit."""
        for (area_i, layer_i, pop_i, cluster_i), target_pop in self.pops.items():
            nest.Connect(self.multimeter, target_pop)

    def get_rates(self):
        """ Reads out the rates from the recording device. """
        rates = {}
        try:
            data = self.multimeter.events
            for (area_i, layer_i, pop_i, cluster_i), gids_i in self.pops.items():
                rates[(area_i, layer_i, pop_i, cluster_i)] = data['rate'][
                    np.where(data['senders'] == gids_i.global_id)
                ]
        except AttributeError:
            data = nest.GetStatus(self.multimeter)[0]['events']
            for (area_i, layer_i, pop_i, cluster_i), gids_i in self.pops.items():
                # 0 bc. only one siegert neuron per population
                rates[(area_i, layer_i, pop_i, cluster_i)] = data['rate'][
                    np.where(data['senders'] == gids_i[0])
                ]
        return pd.DataFrame(rates)

    def solve(self, num_threads):
        """ Execute subfunctions to solve the theory.

        This function executes several subfunctions to create neuronal
        populations, devices and inputs, connects the populations with
        each other and with devices and input nodes.

        Parameters
        ----------
        num_threads : int
        """
        self.setup_nest(num_threads)
        print('Creating neurons and devices')
        self.create_populations()
        self.create_devices()
        self.create_poisson()
        if self.theo_dict['noise_weight'] > 0:
            self.create_noise()
        print('Connecting neurons and devices')
        self.connect_neurons()
        self.connect_poisson()
        if self.theo_dict['noise_weight'] > 0:
            self.connect_noise()
        self.connect_devices()
        print('Simulating')
        nest.Simulate(self.theo_dict['T'])
        return self.get_rates()

    def getHash(self):
        """
        Creates a hash from simulation parameters.

        Returns
        -------
        hash : str
            Hash for the simulation
        """
        hash = dicthash.generate_hash_from_dict(self.theo_dict)
        return hash

    def dump(self, base_folder):
        """
        Exports the full simulation specification. Creates a subdirectory of
        base_folder from the simulation hash where it puts all files.

        Parameters
        ----------
        base_folder : string
            Path to base output folder
        """
        hash = self.getHash()
        out_folder = os.path.join(base_folder, hash)
        try:
            os.mkdir(out_folder)
        except OSError:
            pass

        # output simple data as yaml
        fn = os.path.join(out_folder, 'theo.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.theo_dict, outfile, default_flow_style=False)


def simulationDictFromDump(dump_folder):
    """
    Creates a simulation dict from the files created by Simulation.dump().

    Parameters
    ----------
    dump_folder : string`
        Folder with dumped files

    Returns
    -------
    theo_dict : dict
        Full theory dictionary
    """
    # Read sim.yaml
    fn = os.path.join(dump_folder, 'theo.yaml')
    with open(fn, 'r') as theo_file:
        theo_dict = yaml.load(theo_file)
    return theo_dict


if __name__ == '__main__':
    from network import networkDictFromDump
    from default_theo_params import params as theo_params

    net_hash = '59122d66174ad56ec5520d073dcb9f26'
    net_dict = networkDictFromDump(f'../out/{net_hash}')
    huvi_theory = Theory(theo_params, net_dict)
    rates = huvi_theory.solve(num_threads=1)
    rates = rates.iloc[-1, :]  # only final rates
    print('MFT predictions (spks/s):')
    print(rates.describe())
