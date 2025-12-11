import os
import csv
import copy
import random
from ast import literal_eval
from datetime import datetime
import glob
import math
import time
import subprocess

import yaml
import numpy as np
import pandas as pd
import networkx as nx
# import community as community_louvain

from scipy.io import loadmat
from scipy.signal import convolve, coherence, welch, csd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import seaborn as sns

from dicthash import dicthash

from joblib import Parallel, delayed
from multiprocessing import Pool

from .default_params import complete_area_list
from config import base_path


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        passed_time = round(te - ts, 3)
        print(f'{method.__name__} took {passed_time} s')
        return result
    return timed


class Analysis():
    """
    Class that provides functionality to analyze simulation results.
    """

    def __init__(self, network, simulation, data_list=['spikes'],
                 load_areas=None):
        self.network = network
        self.simulation = simulation
        assert(self.network.label == self.simulation.network.label)

        self.T = self.simulation.T

        self.areas_simulated = self.simulation.areas_simulated
        self.areas_recorded = self.simulation.areas_recorded
        if load_areas is None:
            self.areas_loaded = self.areas_simulated
        else:
            self.areas_loaded = load_areas

        assert(all([area in self.areas_recorded for area in
                    self.areas_loaded])), "Tried to load areas which "
        "were not recorded"

        self.ana_dict = self.network.params['ana_params']
        self.sim_dict = self.simulation.params
        self.net_dict = self.network.params

        self.ana_hash = self.getHash()

        self.sim_folder = self.simulation.data_dir
        # LVD
        self.ana_folder = self.simulation.data_dir
        #self.ana_folder = os.path.join(
        #        self.simulation.data_dir,
        #        self.ana_hash
        #        )
        self.rec_folder = os.path.join(
                self.simulation.data_dir,
                'recordings'
                )

        self.plot_folder = os.path.join(self.ana_folder, 'plots')

        seed = self.ana_dict['seed']
        random.seed(seed)

        if not os.path.isdir(self.ana_folder):
            os.mkdir(self.ana_folder)
        print('Results will be written to %s' % self.plot_folder)
        if not os.path.isdir(self.plot_folder):
            os.mkdir(self.plot_folder)
        print('Plots will be written to %s' % self.plot_folder)

    @timeit
    def fullAnalysis(self):
        """
        Execute the full analysis.
        """
        print('{} Reading popGids'.format(datetime.now().time()))
        self.popGids = self._readPopGids()

        print('{} Reading spikes'.format(datetime.now().time()))
        self.spikes = self._readSpikes()

        print('{} Calculating rate'.format(datetime.now().time()))
        self.rate = self.meanFiringRate()

        print('{} Calculating population CV ISI'.format(datetime.now().time()))
        self.pop_cv_isi = self.popCvIsi()

        print('{} Calculating population LV ISI'.format(datetime.now().time()))
        self.pop_lv = self.popLv()

        print('{} Calculating correlation coefficients.'.format(
            datetime.now().time())
            )
        self.pop_cc = self.popCorrCoeff()

        print('{} Calculating rate histogram'.format(datetime.now().time()))
        self.rate_hist, self.rate_hist_areas = self.firingRateHistogram()

        """print('{} Calculating synaptic input currents'.format(
            datetime.now().time())
        )
        self.curr_in = self.synapticInputCurrent()

        print('{} Calculating coherence'.format(
            datetime.now().time())
        )
        self.coherence_dict = self.compute_coherence()"""

        print('{} Calculating power spectrum'.format(
            datetime.now().time())
        )
        self.power_spectrum_dict = self.compute_power_spectrum()


        """print('{} Calculating BOLD based on synaptic input_current'.format(datetime.now().time()))
        self.BOLD_syn = self.computeBOLD(mode='bold_synaptic_input_current')

        print('{} Calculating BOLD based on rates'.format(datetime.now().time()))
        self.BOLD_rates = self.computeBOLD(mode='rates')

        print('{} Calculating BOLD connectivity based on bold_synaptic_input_current'.format(datetime.now().time()))
        self.BOLD_correlation_syn = self.calculateBOLDConnectivity(mode='bold_synaptic_input_current')

        print('{} Calculating BOLD connectivity based on rates'.format(datetime.now().time()))
        self.BOLD_correlation_rates = self.calculateBOLDConnectivity(mode='rates')

        print('{} Calculating connectivity based on synaptic input currents'.format(datetime.now().time()))
        self.fc_synaptic_currents = self.calculate_fc_based_on_synaptic_currents()

        print('{} Reading in experimental function connectivity'.format(datetime.now().time()))
        self.exp_FC = self.read_in_experimental_functional_connectivity()

        print('{} Calculating correlations between experimental and simulated functional connectivities'.format(datetime.now().time()))
        self.calculate_fc_correlations()

        print('{} Read in experimental spikes'.format(datetime.now().time()))
        self.exp_spikes = self.read_in_experimental_spikes()
        self.write_out_exp_and_simulated_cv()


        # print('{} Calculating connectivity correlations'.format(datetime.now().time()))
        # self.calculateFuncionalConnectivityCorrelations()

        print('{} Plotting {}'.format(datetime.now().time(), 'Boxplot'))
        self.plotBoxPlot()"""

        print('{} Plotting {}'.format(datetime.now().time(), 'Cv Isi'))
        self.plotPopCVIsi()

        print('{} Plotting {}'.format(datetime.now().time(), 'Lv Isi'))
        self.plotPopLV()

        print('{} Plotting {}'.format(
            datetime.now().time(), 'Correlation coefficients')
            )
        self.plotCorrCoff()

        print('{} Plotting {}'.format(
            datetime.now().time(), 'area averaged spike rates cluster resolved')
            )
        self.plotAllFiringRatesSummary_cluster_resolved()
        self.plotAllFiringRatesSummary_cluster_resolved_area_sorted()

        print('{} Plotting {}'.format(
            datetime.now().time(), 'area averaged spike rates')
            )
        self.plotAllFiringRatesSummary()

        """print('{} Plotting {} based on synaptic_input_current'.format(
            datetime.now().time(), 'signal')
            )
        self.plot_synaptic_current_SignalSummary()

        print('{} Plotting {} based on bold_synaptic_input_current'.format(
            datetime.now().time(), 'BOLD signal')
            )
        self.plotAllBOLDSignalSummary(mode='bold_synaptic_input_current')

        print('{} Plotting {} based on rates'.format(
            datetime.now().time(), 'BOLD signal')
            )
        self.plotAllBOLDSignalSummary(mode='rates')

        print('{} Plotting {}'.format(
            datetime.now().time(), 'coherence')
            )
        self.plot_coherence()"""

        print('{} Plotting {}'.format(
            datetime.now().time(), 'power spectrum')
            )
        self.plot_power_spectrum()

        for area in self.popGids.index.levels[0]:
            print('{} Plotting {}'.format(datetime.now().time(), area))
            self.plotRasterArea(area)

        try:
            # print('{} Plotting connectivities'.format(datetime.now().time()))
            # self.plotConnectivities()
            """print('{} Plotting FC connectivities based on BOLD based bold_synaptic_input_current'.format(datetime.now().time()))
            self.plotBOLDConnectivity(mode='bold_synaptic_input_current')

            print('{} Plotting FC connectivities based on BOLD based on rates'.format(datetime.now().time()))
            self.plotBOLDConnectivity(mode='rates')

            print('{} Plotting FC connectivities based on synaptic input currents'.format(datetime.now().time()))
            self.plotBOLDConnectivity(mode='synaptic_input_current')"""
        except OSError:
            print("Something went wrong when plotting community structures."
                  " Probably infomap is not installed.")

    @timeit
    def meanFiringRate(self, group=['area', 'layer', 'pop']):
        """
        Calculates the population averaged firing rate.

        Returns
        -------
        rate : Series
        """
        try:
            rate = pd.read_pickle(os.path.join(self.ana_folder, 'rates.pkl'))
        except FileNotFoundError:
            # Simulation time in seconds
            t_sim_sec = self.sim_dict['t_sim'] / 1000.
            # For all populations we count all spikes that have been emitted
            # during the simulation. Then we divide the total number of spikes
            # per population by the number of neurons in this populations and
            # the simulation time.
            # Group spikes by level which we are interested in
            spikes_tmp = self.spikes.groupby(group).apply(lambda x: [z for y in x.values for z in y])
            # Perform calculation
            rate = spikes_tmp.apply(
                    lambda sts: sum([st.size for st in sts])
                    ).div(self.popGids['recorded_pop_size'].groupby(group).sum()) / t_sim_sec
            rate.to_pickle(os.path.join(self.ana_folder, 'rates.pkl'))
        return rate

    @timeit
    def firingRateHistogram(self, group=['area', 'layer', 'pop']):
        """
        Calculates the time-resolved population averaged firing rate.
        Uses np.histogram with a fixed binsize. Returns the firing rate in
        spikes / [ms].

        Returns
        -------
        rate_hist : Series of numpy arrays containing time resolved firing
                    rates
        """
        fn = 'rate_histogram'
        for x in group:
            fn += '_'
            fn += x
        fn += '.pkl'
        fn_a = 'rate_histogram_areas.pkl'
        try:
            rate_hist = pd.read_pickle(os.path.join(
                self.ana_folder, fn
            ))
            rate_hist_areas = pd.read_pickle(os.path.join(
                self.ana_folder, fn_a
            ))
        except FileNotFoundError:
            # Group spikes by level which we are interested in
            spikes_tmp = self.spikes.groupby(group).apply(lambda x: [z for y in x.values for z in y])
            # Perform calculation
            rate_hist = spikes_tmp.apply(
                    calc_rates,
                    args=(self.sim_dict, self.ana_dict,)
                    ).div(self.popGids.groupby(group).sum()['recorded_pop_size'])

            rate_hist_areas = self.spikes.apply(list).groupby('area').agg(
                    sum
                    ).apply(np.array).apply(
                        calc_rates,
                        args=(self.sim_dict, self.ana_dict)
                    ).div(self.popGids.groupby('area').agg(sum)['recorded_pop_size'])

            rate_hist.to_pickle(os.path.join(
                self.ana_folder, fn
            ))

            rate_hist_areas.to_pickle(os.path.join(
                self.ana_folder, fn_a
            ))
        return rate_hist, rate_hist_areas

    @timeit
    def compute_power_spectrum(self):
        """
        Done for all neurons. In Mam it's only for 140 neurons.
        """
        CLUSTERS = list(self.spikes.index.levels[3])
        CLUSTERS += ['all']

        d = {
                'V1': {},
                'V4': {},
                'FEF': {},
                'LIP': {}
                }

        firing_rates_area, _ = self.firingRateHistogram(['area'])
        firing_rates_cluster, _ = self.firingRateHistogram(['area', 'cluster'])

        for area in d.keys():
            for cluster in CLUSTERS:
                d[area][cluster] = {}
                area_loc = area

                if cluster == 'all':
                    firing_rates = firing_rates_area
                else:
                    firing_rates = firing_rates_cluster
                    area_loc = (area, cluster)

                fr = firing_rates.loc[area_loc]

                if len(fr) > 1024:
                    noverlap = 1000
                    nperseg = 1024
                else:
                    noverlap = None
                    nperseg = None
                f, power = welch(
                        fr,
                        fs=1.e3,  # convert from 1/ms to 1/s
                        noverlap=noverlap,
                        nperseg=nperseg,
                        )
                d[area][cluster]['f'] = f
                d[area][cluster]['power'] = power

        pd.DataFrame(d).to_pickle(
                os.path.join(
                    self.ana_folder,
                    'power_spectrum.pkl'
                    )
                )
        return d

    def plot_power_spectrum(self):
        for key_1, d_1 in self.power_spectrum_dict.items():
            for key_2, d_2 in d_1.items():
                key = key_1 + '_' + key_2
                fig = plt.figure(1)
                plt.semilogy(d_2['f'], d_2['power'])
                plt.xlim(0,100)
                plt.title(key)
                plt.ylabel('Power Spectrum')
                plt.xlabel('Frequency in Hz')
                fig.savefig(os.path.join(
                    self.plot_folder,
                    f'power_spectrum_{key}.png'
                ))
                plt.clf()
                plt.close(fig)

    @timeit
    def compute_coherence(self):
        """
        Calculates the coherences:
        FEF spikes - V4 LFP
        V4 spikes - FEF LFP
        FEF LFP - V4 LFP
        """
        CLUSTERS = list(self.spikes.index.levels[3])
        CLUSTERS += ['all']

        d = {
                'FEF_spikes_V4_LFP': {},
                'V4_spikes_FEF_LFP': {},
                'FEF_LFP_V4_LFP': {},
                'FEF_spikes_FEF_LFP': {},
                'V4_spikes_V4_LFP': {},
                'LIP_spikes_LIP_LFP': {},
                'V4_spikes_V4_spikes': {},
                'FEF_LFP_LIP_LFP': {},
                'LIP_spikes_MT_spikes': {},
                'LIP_LFP_MT_LFP': {},
                }

        firing_rates_area, _ = self.firingRateHistogram(['area'])
        lfp_area = self.synapticInputCurrent(['area'])

        firing_rates_cluster, _ = self.firingRateHistogram(['area', 'cluster'])
        lfp_cluster = self.synapticInputCurrent(['area', 'cluster'])

        for key in d.keys():
            for cluster in CLUSTERS:
                d[key][cluster] = {}

                area_1, method_1, area_2, method_2 = key.split('_')
                if cluster == 'all':
                    firing_rates = firing_rates_area
                    lfp = lfp_area
                else:
                    firing_rates = firing_rates_cluster
                    lfp = lfp_cluster
                    area_1 = (area_1, cluster)
                    area_2 = (area_2, cluster)

                if method_1 == 'spikes':
                    series_1 = firing_rates.loc[area_1][500:]
                else:
                    series_1 = lfp.loc[area_1][500:]

                if method_2 == 'spikes':
                    series_2 = firing_rates.loc[area_2][500:]
                else:
                    series_2 = lfp.loc[area_2][500:]

                if len(series_1) > 1024:
                    noverlap = 1000
                    nperseg = 1024
                else:
                    noverlap = None
                    nperseg = None

                Pxx_list = []
                Pyy_list = []
                Pxy_list = []
                # Note: 10000 is hardcoded and requires a 10 second simulation
                # TODO fix this
                t_sim = self.sim_dict['t_sim']
                t_upper = int((t_sim - 500)//2000*2000 + 500)
                for i in range(500, t_upper, 2000):
                    f_1, Pxx_ = welch(
                            series_1[i:i+2000],
                            noverlap=noverlap,
                            nperseg=nperseg,
                            fs=1.e3  # convert from 1/ms to 1/s
                            )
                    f_2, Pyy_ = welch(
                            series_2[i:i+2000],
                            noverlap=noverlap,
                            nperseg=nperseg,
                            fs=1.e3  # convert from 1/ms to 1/s
                            )
                    f_3, Pxy_ = csd(
                            series_1[i:i+2000],
                            series_2[i:i+2000],
                            noverlap=noverlap,
                            nperseg=nperseg,
                            fs=1.e3  # convert from 1/ms to 1/s
                            )
                    Pxx_list.append(Pxx_)
                    Pyy_list.append(Pyy_)
                    Pxy_list.append(Pxy_)

                Pxx = np.mean(np.array(Pxx_list), axis=0)
                Pyy = np.mean(np.array(Pyy_list), axis=0)
                Pxy = np.mean(np.array(Pxy_list), axis=0)

                Cxy = np.sqrt(np.abs(Pxy)**2 / ( Pxx * Pyy ))

                assert (f_1 == f_2).all() & (f_1 == f_3).all()

                d[key][cluster]['f'] = f_1
                d[key][cluster]['Cxy'] = Cxy

        pd.DataFrame(d).to_pickle(
                os.path.join(
                    self.ana_folder,
                    'coherences.pkl'
                    )
                )
        return d

    def plot_coherence(self):
        for key_1, d_1 in self.coherence_dict.items():
            for key_2, d_2 in d_1.items():
                key = key_1 + '_' + key_2
                fig = plt.figure(1)
                plt.plot(d_2['f'], d_2['Cxy'])
                plt.xlim(0,100)
                plt.title(key)
                plt.ylabel('Coherence')
                plt.xlabel('Frequency in Hz')
                fig.savefig(os.path.join(
                    self.plot_folder,
                    f'coherence_{key}.png'
                ))
                plt.clf()
                plt.close(fig)

    @timeit
    def synapticInputCurrent(self, group=['area']):
        """
        Calculates the area averaged synaptic input current.
        Compare MAM dynamics paper, page14.

        Returns
        -------
        current : DataFrame (index=index, columns=t)
        """
        fn = 'input_current'
        for x in group:
            fn += '_'
            fn += x
        fn += '.pkl'
        try:
            curr = pd.read_pickle(os.path.join(
                self.ana_folder, fn
            ))
        except FileNotFoundError:
            # Get parameters
            # Note: firingRateHistogram returns the rates in spikes / [ms]. In
            # the original multi-area model the firing rate histogram is used
            # in units 1 / [s]. This leads to a difference of 1000 x in the
            # result of the synaptic current calculation. This is just a
            # numerical difference.
            inst_rates, _ = self.firingRateHistogram(group=['area', 'layer', 'pop', 'cluster'])
            index = inst_rates.index
            NN = pd.read_pickle(os.path.join(self.sim_folder, 'N.pkl')).loc[index]
            weights = pd.read_pickle(os.path.join(self.sim_folder, 'W.pkl')).loc[index, index]
            synapses = pd.read_pickle(os.path.join(self.sim_folder, 'synapses.pkl')).loc[index, index]

            indeg = synapses.div(NN, axis=1)
            tau_s_e = self.net_dict['neuron_params']['single_neuron_dict']['tau_syn_ex']
            tau_s_i = self.net_dict['neuron_params']['single_neuron_dict']['tau_syn_in']
            dt = self.sim_dict['dt']
            t_sim = self.sim_dict['t_sim']
            binsize = self.ana_dict['rate_histogram_binsize']

            # Calculate exponential kernel for PSCs
            # TODO different synaptic time constants
            assert(tau_s_e == tau_s_i)
            tau_s = tau_s_e

            mam_like = True
            if mam_like:
                t_ker = np.arange(0., 20., 1.)  # np.arange(-10*tau_s, 10*tau_s, dt)
                kernel = np.exp(-t_ker / tau_s)  # np.exp(- t_ker / tau_s) / tau_s
            else:
                t_ker = np.arange(-10*tau_s, 10*tau_s, dt)
                kernel = np.exp(- t_ker / tau_s) / tau_s
                kernel[t_ker < -dt/2] = 0  # Make filter causal
                kernel /= kernel.sum()  # Normalize filter

            # Calculate filtered rates
            t_in = np.arange(0, t_sim, binsize)

            use_external_rates = False
            if use_external_rates:
                # Calculate synaptic currents induced by external stimulation
                idx = pd.IndexSlice
                ext_rate = self.network.params['input_params']['rate_ext']  # Rate in spikes / [sec]
                ext_rate /= 1000.  # to spikes / [ms]
                fac_nu_ext_5E = self.network.params['connection_params']['fac_nu_ext_5E']  # Factor
                fac_nu_ext_6E = self.network.params['connection_params']['fac_nu_ext_6E']  # Factor
                fac_nu_ext_TH = self.network.params['connection_params']['fac_nu_ext_TH']  # Factor
                num_rates = inst_rates.apply(len).iloc[0]
                tmp_series = pd.Series(data=num_rates, index=inst_rates.index)
                ext_rates = tmp_series.apply(lambda x: ext_rate * np.ones(x))
                ext_rates.loc[idx[:, '5', 'E']] = ext_rates.loc[idx[:, '5', 'E']].values * fac_nu_ext_5E
                ext_rates.loc[idx[:, '6', 'E']] = ext_rates.loc[idx[:, '6', 'E']].values * fac_nu_ext_6E
                ext_rates.loc[idx['TH', :, :]] = ext_rates.loc[idx['TH', :, :]].values * fac_nu_ext_TH

                pscs_ext = ext_rates.apply(convolve,args=(kernel, 'same',)).apply(pd.Series)
                curr_ext = tau_s * (weights.abs() * indeg).dot(pscs_ext)
                curr_ext = curr_ext.mul(NN, axis=0).groupby(level=0).sum()
                curr_ext = curr_ext.div(NN.groupby(level=0).sum(), axis=0)
                curr_ext = curr_ext.agg(np.array, axis=1)

            # Apply convolution to all numpy arrays stored in a cell. This
            # gives back a single cell with an array containing the results.
            # Thus afterwards apply pd.Series to split the arrays in cells into
            # cells containing only one value.
            pscs = inst_rates.apply(
                    convolve,
                    args=(kernel, 'same',)
                    ).apply(pd.Series)
            pscs.columns = t_in

            delayed_synaptic_current = False
            if delayed_synaptic_current:
                # Include weights, indegrees and average
                v = self.network.params['delay_params']['interarea_speed']
                # s = self.network.distances[self.name][source_area_name]
                c = pd.DataFrame(data=0., index=pscs.index, columns=pscs.columns, dtype=np.float64)
                b = tau_s * weights.abs() * indeg
                # Loop over target areas
                for area, df in b.groupby('area', axis=0):
                    # Now I am in a target area. How are the input currents from other areas to this area?
                    # All delays from all source areas are:
                    s = self.network.distances[area]
                    delays = pd.Series(s)
                    tmp_pscs = pd.DataFrame(data=0., index=pscs.index, columns=pscs.columns, dtype=np.float64)
                    # We loop over all post synaptic currents and apply a shift to
                    # them. The shift corresponds to the delay from source to
                    # target area
                    for tmp_area, tmp in pscs.iterrows():
                        tmp_pscs.loc[tmp_area] = tmp.shift(math.floor(delays[tmp_area[0]]), fill_value=0.)
                    c.loc[area] = df.dot(tmp_pscs)
                curr = c
            else:
                # Include weights, indegrees and average
                # We need to transpose the matrix because we want to calculate
                # the effect of the source onto the target area, not vice versa
                if mam_like:
                    curr = (weights.abs() * indeg).T.dot(pscs)
                else:
                    curr = tau_s * (weights.abs() * indeg).T.dot(pscs)

            # Weigh the contributions to the synaptic currents according to
            # the number of neurons in a population
            curr = curr.mul(NN, axis=0).groupby(group).sum()
            curr = curr.div(NN.groupby(group).sum(), axis=0)
            # Calculate also as array, this is not compatible with all
            # functions atm, TODO think about way around this
            curr = curr.agg(np.array, axis=1)

            if use_external_rates:
                curr += curr_ext
            curr.to_pickle(os.path.join(
                self.ana_folder, fn
            ))
        return curr

    @timeit
    def computeBOLD(self, mode='bold_synaptic_input_current'):
        # TODO
        # Use rate ( note it's given in spikes/ms, i.e. multiply by 1000)
        # instead of current. This is in agreement with Gustavo Deco and Viktor
        # Jirsa 2012: Ongoing Cortical Activity at Rest: Criticality,
        # Multistability,and Ghost Attractors
        # and Deco 2019: Brain songs framework used for discovering the
        # relevant timescale of the human brain
        # Kringelbach et al. Dynamic coupling of whole-brain neuronal
        # andneurotransmitter systems
        # It seems, at least in deco 2019 and kringelbach 2020, they only used
        # excitatory neurons
        # Take parameters from Stephan 2017
        try:
            BOLDSIGNAL = pd.read_pickle(os.path.join(
                self.ana_folder, f'bold_signal_{mode}.pkl'
            ))
        except FileNotFoundError:
            if mode == 'rates':
                if not hasattr(self, 'rate_hist_areas'):
                    _, self.rate_hist_areas = self.firingRateHistogram()
                base_signal = self.rate_hist_areas * 1000.  # convert
                                                            # spikes / [ms] to
                                                            # spikes / [s]
                # base_signal = base_signal.apply(lambda x: x[500:])
            elif mode == 'bold_synaptic_input_current':
                if not hasattr(self, 'self.curr_in'):
                    self.curr_in = self.synapticInputCurrent()
                # fn = '/p/scratch/cjinb33/jinb3330/mattention_bold/98e2cf8e390d007136b6f99c31d95d22/eafe8d385d61a3ab2fde17ee1cbea3ad/input_current.pkl'
                # base_signal_ground_state = pd.read_pickle(fn)
                # mean_ground_state = base_signal_ground_state.apply(np.mean)
                base_signal = self.curr_in
                # base_signal = self.curr_in * 1000.  # convert
                                                    # current / [ms] to
                                                    # current / [s]
                # base_signal = base_signal.apply(lambda x: x[500:])
                # We need to normalize the synaptic currents. See for example
                # bonaiuto2014 et al., 2014. The question is:
                # How do we normalize?
                # All other values are normalized (Buxton 1998)
                base_signal_mean = base_signal.apply(np.mean).mean()
                # base_signal_mean = base_signal.apply(np.mean).mean() / 32
                # base_signal_mean = base_signal.apply(lambda x: np.mean(x[500:1000])).mean()
                # base_signal_mean = np.mean(sum(base_signal.values))
                # base_signal_mean = base_signal.apply(np.mean)
                # base_signal = (base_signal - base_signal_mean) / base_signal_mean
                # base_signal = base_signal / base_signal_mean
                base_signal = (base_signal - base_signal.apply(np.mean)) / base_signal_mean
                # base_signal = (base_signal - mean_ground_state) / base_signal_mean
                # base_signal = (base_signal - 3*mean_ground_state) / base_signal_mean

            pool = Pool()
            bold = pool.map(calculate_BOLD, base_signal.items())

            tmp = {}
            for area, b, t in bold:
                tmp[(area, 'bold')] = b
                tmp[(area, 't')] = t

            BOLDSIGNAL = pd.Series(tmp)
            BOLDSIGNAL.to_pickle(os.path.join(
                self.ana_folder, f'bold_signal_{mode}.pkl'
            ))
        return BOLDSIGNAL

    def plotBoxPlot(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.rate'):
            self.rate = self.meanFiringRate()
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        area = np.unique(self.rate.index.get_level_values(0))
        layer = np.unique(self.rate.index.get_level_values(1))
        pop_type = np.unique(self.rate.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        rates = pd.DataFrame(data=0, index=area, columns=ind)

        for (a, l, p), r in self.rate.items():
            rates.loc[a, l+p] = r
        ax = sns.boxplot(
            data=rates,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i in range(len(ind)):
            mybox = ax.artists[i]
            mybox.set_facecolor(col[i % 2])
        plt.xlabel('Rate (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'boxplot.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    @timeit
    def popCorrCoeff(self, group=['area', 'layer', 'pop']):
        """
        Compute correlation coefficients for a subsample of neurons for the
        entire network. Subsample is set to 200 for duration reasons. Higher
        subsample size might yield more accurate results but is costly timing
        wise.

        Returns
        -------
        mean : Pandas Dataframe
            Population-averaged correlation coefficients.
        """
        try:
            cc = pd.read_pickle(os.path.join(
                self.ana_folder, 'cc.pkl'
            ))
        except FileNotFoundError:
            # Group spikes by level which we are interested in
            spikes_tmp = self.spikes.groupby(group).apply(lambda x: [z for y in x.values for z in y])
            # Perform calculation
            # cc = Parallel(n_jobs=self.sim_dict['analysis']['local_num_threads'], prefer="threads")(
            cc = Parallel(n_jobs=1)(
                delayed(correlation)(
                    sts,
                    self.ana_dict
                    ) for sts in spikes_tmp.values
            )
            cc = pd.Series(cc, index=spikes_tmp.index)

            cc.to_pickle(os.path.join(self.ana_folder, 'cc.pkl'))
        return cc

    def plotCorrCoff(self):
        """
        Generates a boxplot of the correlation coefficients of the different
        populations averaged over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_cc'):
            self.pop_cc = self.popCorrCoeff()
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        area = np.unique(self.pop_cc.index.get_level_values(0))
        layer = np.unique(self.pop_cc.index.get_level_values(1))
        pop_type = np.unique(self.pop_cc.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_ccs = pd.DataFrame(data=0, index=area, columns=ind)

        for (a, l, p), r in self.pop_cc.items():
            pop_ccs.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_ccs,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i, patch in enumerate(ax.patches):
            patch.set_facecolor(col[i % 2])
        if len(ind) != len(ax.patches):
            print(f"WARNING: not all areas contain data")
        plt.xlabel('Correlation coefficient')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'cc.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    @timeit
    def popLv(self, group=['area', 'layer', 'pop']):
        """
        Compute the Lv value of the spikes.
        See Shinomoto et al. 2009 for details.

        Returns
        -------
        mean : Pandas Dataframe
            Population-averaged Lv.
        """
        try:
            lv = pd.read_pickle(os.path.join(
                self.ana_folder, 'lv.pkl'
            ))
        except FileNotFoundError:
            # Question: divide by calculated lvs or all neurons?
            # Answer: in cv and lv divide by the number of neurons that have
            # actually spiked,
            t_start = self.ana_dict['lvr']['t_start']
            t_stop = self.ana_dict['lvr']['t_stop']
            t_ref = self.net_dict['neuron_params']['single_neuron_dict']['t_ref']
            # Group spikes by level which we are interested in
            spikes_tmp = self.spikes.groupby(group).apply(lambda x: [z for y in x.values for z in y])
            # Perform calculation
            lv = spikes_tmp.apply(LV, args=(t_ref, t_start, t_stop))
            lv.to_pickle(os.path.join(self.ana_folder, 'lv.pkl'))
        return lv

    # def compute_louvain(self):
    #     """
    #     Calculates the louvain clustering of simulated area averaged synaptic
    #     input current.

    #     Returns
    #     -------
    #     mean : Pandas Dataframe
    #         Population-averaged Lv.
    #     funct_conn_sorted :
    #         Calculated functional connectivity sorted with respect to
    #         clustering
    #     areas_parted : list
    #         Sorted area list with full area names
    #     areas_parted_short : list
    #         Sorted area list with abbreviated area names
    #     louvain_cluster : dict
    #         Found Louvain clustering as a dict
    #     df_louvain_cluster : Series
    #         Found Louvain clustering as a pandas series
    #     """
    #     # TODO What to return, where to call
    #     if not hasattr(self, 'self.rate'):
    #         self.curr_in = self.synapticInputCurrent()

    #     tmin = self.ana_dict['compute_louvain']['tmin']
    #     area_list = self.curr_in.index.values

    #     # Normalize calculate func_conn
    #     bold = self.curr_in.T
    #     funct_conn = bold[bold.index >= tmin].corr()

    #     G = nx.Graph()
    #     for area in area_list:
    #         G.add_node(area)
    #     edges = []
    #     for area1 in area_list:
    #         for area2 in area_list:
    #             edges.append((area1, area2, funct_conn.loc[area1, area2]))
    #     G.add_weighted_edges_from(edges)

    #     louvain_cluster = community_louvain.best_partition(G)
    #     df_louvain_cluster = pd.Series(louvain_cluster)

    #     areas_parted = df_louvain_cluster.sort_values().index.values
    #     areas_parted_short = [dk_full_to_short[name] for name in areas_parted]
    #     funct_conn_sorted = funct_conn.loc[areas_parted, areas_parted]

    #     louvain_cluster.to_pickle(
    #             os.path.join(self.sim_folder, 'louvain_clustering.pkl')
    #             )
    #     return funct_conn_sorted, areas_parted, areas_parted_short, louvain_cluster, df_louvain_cluster

    def plotPopLV(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_lv'):
            self.pop_lv = self.popLv()
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        area = np.unique(self.pop_lv.index.get_level_values(0))
        layer = np.unique(self.pop_lv.index.get_level_values(1))
        pop_type = np.unique(self.pop_lv.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_lvs = pd.DataFrame(data=0, index=area, columns=ind)

        for (a, l, p), r in self.pop_lv.items():
            pop_lvs.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_lvs,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i, patch in enumerate(ax.patches):
            patch.set_facecolor(col[i % 2])
        if len(ind) != len(ax.patches):
            print(f"WARNING: not all areas contain data")
        plt.xlabel('Lv (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'lv_isi.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    @timeit
    def popCvIsi(self, group=['area', 'layer', 'pop']):
        """
        Calculates the coefficient of variation cv of the interspike intervals
        isi.

        Returns
        -------
        mean : Pandas Dataframe
            Mean CV ISI value of the population
        """
        try:
            cv_isi = pd.read_pickle(os.path.join(
                self.ana_folder, 'cv_isi.pkl'
            ))
        except FileNotFoundError:
            # Question: divide by calculated cv isis or all neurons?
            # Answer: in cv and lv divide by the number of neurons that have
            # actually spiked,
            t_start = self.ana_dict['cv']['t_start']
            t_stop = self.ana_dict['cv']['t_stop']
            # Group spikes by level which we are interested in
            spikes_tmp = self.spikes.groupby(group).apply(lambda x: [z for y in x.values for z in y])
            # Perform calculation
            cv_isi = spikes_tmp.apply(cvIsi, args=(t_start, t_stop))
            cv_isi.to_pickle(os.path.join(self.ana_folder, 'cv_isi.pkl'))
        return cv_isi

    def plotPopCVIsi(self):
        """
        Generates a boxplot of the rates of the different populations averaged
        over all areas.
        ----------
        """
        if not hasattr(self, 'self.pop_cv_isi'):
            self.pop_cv_isi = self.popCvIsi()
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        area = np.unique(self.pop_cv_isi.index.get_level_values(0))
        layer = np.unique(self.pop_cv_isi.index.get_level_values(1))
        pop_type = np.unique(self.pop_cv_isi.index.get_level_values(2))

        multi_index = pd.MultiIndex.from_product([layer, pop_type])
        ind = [''.join(i) for i in multi_index.tolist()]

        pop_cv_isis = pd.DataFrame(data=0, index=area, columns=ind)

        for (a, l, p), r in self.pop_cv_isi.items():
            pop_cv_isis.loc[a, l+p] = r
        ax = sns.boxplot(
            data=pop_cv_isis,
            orient="h",
            palette="Set2",
        )
        col = ['blue', 'red']
        for i, patch in enumerate(ax.patches):
            patch.set_facecolor(col[i % 2])
        if len(ind) != len(ax.patches):
            print(f"WARNING: not all areas contain data")
        plt.xlabel('Cv Isi (spikes/s)')
        plt.ylabel('Population')
        fig.savefig(os.path.join(
            self.plot_folder,
            'cv_isi.{0}'.format(extension)
        ))
        plt.clf()
        plt.close(fig)

    # def plotConnectivities(self):
    #     """
    #     Plots structural and functional connectivities
    #     """
    #     # Read values
    #     if not hasattr(self, 'self.rate'):
    #         self.curr_in = self.synapticInputCurrent()
    #     extension = self.ana_dict['extension']
    #     tmin = self.ana_dict['plotConnectivities']['tmin']
    #     bold = self.curr_in.T
    #     synapses = self.net_dict['synapses_internal']
    #     # Aggregate synapses to area-level
    #     synapses = synapses.groupby(level=0).sum().T.groupby(level=0).sum().T

    #     # Normalize struct_conn, calculate func_conn
    #     struct_conn = synapses
    #     struct_conn[struct_conn < 1] = np.NaN
    #     struct_conn = np.log10(struct_conn)
    #     funct_conn = bold[bold.index >= tmin].corr()
    #     np.fill_diagonal(funct_conn.values, np.NaN)

    #     # Map area names to short area names & add cluster membership to index
    #     funct_conn.index = [
    #         self.cluster_membership_Synapses,
    #         funct_conn.index.map(dk_full_to_short)
    #     ]
    #     funct_conn.index.names = ['', '']
    #     funct_conn.columns = [
    #         self.cluster_membership_Synapses,
    #         funct_conn.columns.map(dk_full_to_short)
    #     ]
    #     funct_conn.columns.names = ['', '']
    #     struct_conn.index = [
    #         self.cluster_membership_Synapses,
    #         struct_conn.index.map(dk_full_to_short)
    #     ]
    #     struct_conn.index.names = ['', '']
    #     struct_conn.columns = [
    #         self.cluster_membership_Synapses,
    #         struct_conn.columns.map(dk_full_to_short)
    #     ]
    #     struct_conn.columns.names = ['', '']

    #     # Sort according to cluster membership
    #     funct_conn = funct_conn.sort_index(axis=0).sort_index(axis=1)
    #     struct_conn = struct_conn.sort_index(axis=0).sort_index(axis=1)

    #     # Plot using seaborn heatmap
    #     fig, axarr = plt.subplots(1, 2, figsize=(12, 6))
    #     axarr[0].set_title('Structural connectivity (log synapse number)')
    #     sns.heatmap(
    #         struct_conn, square=True, ax=axarr[0], cmap="YlGnBu",
    #         cbar=True, cbar_kws={"shrink": .66}
    #     )
    #     axarr[1].set_title('Functional connectivity (abs. input currents)')
    #     mask = np.zeros_like(funct_conn)
    #     mask[np.triu_indices_from(mask)] = True
    #     sns.heatmap(
    #         funct_conn, square=True, ax=axarr[1], mask=mask, cmap="YlGnBu",
    #         cbar=True, cbar_kws={"shrink": .66}
    #     )
    #     plt.tight_layout()
    #     fig.savefig(os.path.join(
    #         self.plot_folder,
    #         'connectivity.{0}'.format(extension)
    #     ))
    #     plt.clf()
    #     plt.close(fig)

    @timeit
    def calculate_fc_based_on_synaptic_currents(self):
        if not hasattr(self, 'self.curr_in'):
            self.curr_in = self.synapticInputCurrent()
        correlations = pd.DataFrame(
                data=0,
                index=self.curr_in.index,
                columns=self.curr_in.index,
                dtype=np.float64
                )
        for source_area, source_current in self.curr_in.items():
            for target_area, target_current in self.curr_in.items():
                if source_area != target_area:
                    tmp = np.corrcoef(source_current, target_current)[0, 1]
                    correlations.loc[source_area][target_area] = tmp
        correlations.to_pickle(
                os.path.join(
                    self.ana_folder,
                    f'functional_connectivity_synaptic_input_currents.pkl'
                    )
                )
        return correlations



    @timeit
    def calculateBOLDConnectivity(self, mode='bold_synaptic_input_current'):
        """
        Calculates BOLD connectivities
        """
        try:
            BOLD_correlation = pd.read_pickle(
                    os.path.join(self.ana_folder, f'bold_correlation_{mode}.pkl')
                    )
        except FileNotFoundError:
            if mode == 'bold_synaptic_input_current':
                if not hasattr(self, 'self.BOLD_syn'):
                    self.BOLD_syn = self.computeBOLD(mode=mode)
                base_signal = self.BOLD_syn
            elif mode == 'rates':
                if not hasattr(self, 'self.BOLD_rates'):
                    self.BOLD_rates = self.computeBOLD(mode=mode)
                base_signal = self.BOLD_rates
            # Start and end time, cast to [s]
            t_min = self.ana_dict['plotBOLD']['tmin'] / 1000.
            t_max = self.sim_dict['t_sim'] / 1000.

            # The BOLD output was calculated by rk45, which does not give the
            # results on a specified grid. Thus we need to sample all signals
            # to the same grid.
            bold_timesteps = self.ana_dict['plotBOLD']['stepSize'] / 1000.
            t_new = np.arange(
                    t_min,
                    t_max,
                    bold_timesteps
                    )

            BOLD_RESAMPLED = base_signal.copy()
            for area, data in base_signal.groupby(level=0):
                t = data.loc[area, 't']
                b = data.loc[area, 'bold']
                # Last value might be out of range, extrapolate it
                bold_fun = interp1d(
                        t,
                        b,
                        bounds_error=False,
                        fill_value='extrapolate'
                        )
                b_new = bold_fun(t_new)
                BOLD_RESAMPLED.loc[area, 't'] = t_new
                BOLD_RESAMPLED.loc[area, 'bold'] = b_new

            # Calculate the correlations
            index = BOLD_RESAMPLED.loc[:, 'bold'].index
            BOLD_correlation = pd.DataFrame(
                    index=index,
                    columns=index,
                    dtype=np.float64
                    )
            for source in index:
                for target in index:
                    if source != target:
                        BOLD_correlation.loc[source, target] = np.corrcoef(
                                [
                                    BOLD_RESAMPLED.loc[source, 'bold'],
                                    BOLD_RESAMPLED.loc[target, 'bold']
                                    ]
                                )[0,1]
            BOLD_correlation.index.name = 'area'
            BOLD_correlation.columns.name = 'area'

            BOLD_correlation.to_pickle(
                    os.path.join(self.ana_folder, f'bold_correlation_{mode}.pkl')
                    )
        return BOLD_correlation

    # @timeit
    # def calculateFuncionalConnectivityCorrelations(self):
    #     if not hasattr(self, 'self.BOLD_correlation'):
    #         self.BOLD_correlation = self.calculateBOLDConnectivity()
    #     if not hasattr(self, 'self.rate'):
    #         self.curr_in = self.synapticInputCurrent()
    #     outfile = open(os.path.join(self.ana_folder, 'correlations.txt'), 'w')
    #     tmin = self.ana_dict['plotConnectivities']['tmin']
    #     funct_conn = self.curr_in.T

    #     # Normalize struct_conn, calculate func_conn
    #     funct_conn = funct_conn[funct_conn.index >= tmin].corr()
    #     np.fill_diagonal(funct_conn.values, np.NaN)

    #     # Sort
    #     funct_conn = funct_conn.sort_index(axis=0).sort_index(axis=1)
    #     BOLD_correlation = self.BOLD_correlation.sort_index(axis=0).sort_index(
    #             axis=1
    #             )

    #     bold_vals = BOLD_correlation.values.ravel()
    #     bold_vals = bold_vals[~np.isnan(bold_vals)]
    #     funct_vals = funct_conn.values.ravel()
    #     funct_vals = funct_vals[~np.isnan(funct_vals)]
    #     correlation_bold_functional = np.corrcoef([bold_vals, funct_vals])[0,1]

    #     # This is PEP 8 conform, but is it really nice?
    #     outfile.write(
    #             ('correlation_simulated_bold_simulated_functional_connectivity'
    #             f': {correlation_bold_functional}')
    #             )
    #     outfile.close()

    def plotBOLDConnectivity(self, mode='bold_synaptic_input_current'):
        if mode == 'bold_synaptic_input_current':
            if not hasattr(self, 'BOLD_correlation_syn'):
                self.BOLD_correlation_syn = self.calculateBOLDConnectivity(mode=mode)
            base_signal = self.BOLD_correlation_syn
        elif mode == 'rates':
            if not hasattr(self, 'BOLD_correlation_rates'):
                self.BOLD_correlation_rates = self.calculateBOLDConnectivity(mode=mode)
            base_signal = self.BOLD_correlation_rates
        if mode == 'synaptic_input_current':
            if not hasattr(self, 'fc_synaptic_currents'):
                self.fc_synaptic_currents = self.calculate_fc_based_on_synaptic_currents()
            base_signal = self.fc_synaptic_currents
        # Same ordering as in fig 8 in schmidt et al.
        ordering = ['V1', 'V2', 'VP', 'V4t', 'V4', 'VOT', 'MSTd', 'PITv', 'PITd', 'CITv', 'CITd', 'AITv', 'AITd', 'MDP', 'V3', 'V3A', 'MT', 'PIP', 'PO', 'DP', 'MIP', 'VIP', 'LIP', 'MSTl', 'FEF', 'TF', 'FST', '7a', 'STPp', 'STPa', '46', 'TH']
        custom_cmap = plt.cm.coolwarm
        myblue = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)
        myred = (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)
        custom_cmap = custom_cmap.from_list('mycmap', [myblue, 'white', myred], N=256)
        # Plot
        extension = self.ana_dict['extension']
        fig = plt.figure(1)
        sns.heatmap(
                base_signal.loc[ordering][ordering],
                square=True,
                cmap=custom_cmap,  # "YlGnBu",
                cbar=True,
                cbar_kws={"shrink": .66},
                vmin=-1.,
                vmax=1.
                )
        plt.tight_layout()
        fig.savefig(os.path.join(
            self.plot_folder,
            f'bold_correlation_{mode}.{extension}'
        ))
        plt.clf()
        plt.close(fig)

    def plotRasterArea(self, area, group=['area', 'layer', 'pop', 'cluster'], begin=None, end=None):
        """
        Generates a rasterplot of the spiking activity in a specified area.
        Parameters. low and high are in ms.
        ----------
        area : string
        """
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        if not hasattr(self, 'spikes'):
            self.spikes = self._readSpikes()
        extension = self.ana_dict['extension']
        fraction = self.ana_dict['plotRasterArea']['fraction']
        if pd.isnull(begin):
            low = self.ana_dict['plotRasterArea']['low']
        else:
            low = begin
        if pd.isnull(end):
            high = self.ana_dict['plotRasterArea']['high']
        else:
            high = end
        fig = plt.figure(1)
        ax1 = fig.add_subplot(1, 1, 1)
        ind = []
        names = []
        gid_norm = 0
        ms_to_s = 0.001
        colors = {'E': '#1f77b4', 'I': '#ff7f0e'}
        spikes_tmp = self.spikes.groupby(group).apply(lambda x: [z for y in x.values for z in y])
        num_clusters = int(spikes_tmp.loc[area].index[-1][-1])
        for (layer, pop, cluster), sts in spikes_tmp.loc[area].items():
            # Random shuffle spiketrains in place
            random.shuffle(sts)
            j = 0
            # Real recorded population size (accounts for Nrec_spikes_fraction)
            pop_size = self.popGids['recorded_pop_size'][area, layer, pop, cluster]
            pop_size_without_cluster = self.popGids.groupby(['area', 'layer', 'pop']).agg("sum")['recorded_pop_size'][area, layer, pop]

            mini_cluster = \
                self.network.params['connection_params']['mini_cluster']
            # Fraction of total number of neurons
            if mini_cluster and cluster != '0':
                no_sts = int(pop_size)
            else:
                no_sts = int(fraction * pop_size)
            # Fraction of neurons that actually spiked
            frac_spiking = len(sts) * 1. / pop_size

            if cluster == '0':
                # y label axis naming
                name = ' '.join([layer, pop])
                names.append(name)
                # where to put y label
                ind.append(- int(fraction * pop_size_without_cluster / 2) + gid_norm)

            # Loop as many times as we have spike trains
            for _ in range(no_sts):
                gid_norm = gid_norm - 1
                # Decide whether spiketrain contains spikes
                if random.random() < frac_spiking and j < len(sts):
                    st = sts[j]
                    j += 1
                    filtered_st = st[st > low]
                    filtered_st = filtered_st[filtered_st < high]
                    # TODO beautify plot
                    if len(filtered_st) > 0:
                        ax1.plot(
                            filtered_st * ms_to_s,
                            gid_norm * np.ones_like(filtered_st),
                            colors[pop],
                            marker='.',
                            markersize=2,
                            linestyle="None"
                        )
            ax1.axhline(gid_norm, color='grey', linewidth=.5)

        ax1.axis([low * ms_to_s, high * ms_to_s, gid_norm, 0])
        ax1.set_xlabel('Time (s)')
        ax1.set_yticks(ind)
        ax1.set_yticklabels(names)
        ax1.set_title(area)
        fig.savefig(os.path.join(
            self.plot_folder,
            f'raster_{area}_{low}_{high}.{extension}'
        ))
        plt.clf()
        plt.close(fig)

    def plotAllBOLDSignalSummary(self, mode='bold_synaptic_input_current'):
        """
        Generates a summary plot giving an overview over the BOLD signal
        in the network.
        """
        if mode == 'bold_synaptic_input_current':
            if not hasattr(self, 'self.BOLD_syn'):
                self.BOLD_syn = self.computeBOLD(mode=mode)
            base_signal = self.BOLD_syn
        elif mode == 'rates':
            if not hasattr(self, 'self.BOLD_rates'):
                self.BOLD_rates = self.computeBOLD(mode=mode)
            base_signal = self.BOLD_rates
        # Determine max bold for y axis
        max_bold = base_signal[:, 'bold'].apply(max).max()
        min_bold = base_signal[:, 'bold'].apply(min).min()

        min_time = 0.
        max_time = self.sim_dict['t_sim'] / 1000.

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllBOLDSignalSummary']['num_col']
        num_row = math.ceil(len(base_signal[:, 'bold']) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-darkgrid')
        fig, axes = plt.subplots(
                num_row,
                num_col,
                figsize=(22, 22)
                )

        # multiple line plot
        for num, (area, (data, times)) in enumerate(base_signal.groupby(level=0)):
            # Find the right spot on the plot
            ax = axes[num // num_col][num % num_col]

            # Plot the lineplot
            ax.plot(
                    times,
                    data,
                    marker='',
                    color='black',
                    linewidth=1.9,
                    alpha=0.9,
                    )

            # Same limits for everybody!
            ax.set_xlim(min_time, max_time)
            ax.set_ylim(min_bold, max_bold)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('BOLD')

            # Add title
            ax.set_title(
                    area,
                    loc='left',
                    fontsize=20,
                    fontweight=0,
                    color='black'
                    )

        fig.tight_layout()
        plt.savefig(os.path.join(
            self.plot_folder,
            f'bold_overview_{mode}.png'
            ))
        plt.clf()
        plt.close(fig)
        plt.style.use('default')

    def plot_synaptic_current_SignalSummary(self):
        """
        Generates a summary plot giving an overview over the BOLD signal
        in the network.
        """
        if not hasattr(self, 'self.curr_in'):
            self.curr_in = self.synapticInputCurrent()
        base_signal = self.curr_in
        # Determine max bold for y axis
        max_bold = base_signal.apply(max).max()
        min_bold = base_signal.apply(min).min()
        # Set some number for number of columns
        num_col = self.ana_dict['plotAllBOLDSignalSummary']['num_col']
        num_row = math.ceil(len(base_signal) / num_col)

        t_sim = self.sim_dict['t_sim']
        binsize = self.ana_dict['rate_histogram_binsize']
        times = np.arange(0, t_sim, binsize) / 1000.
        # Initialize the figure
        plt.style.use('seaborn-darkgrid')
        fig, axes = plt.subplots(
                num_row,
                num_col,
                figsize=(22, 22)
                )

        # multiple line plot
        for num, (area, data) in enumerate(base_signal.items()):
            # Find the right spot on the plot
            ax = axes[num // num_col][num % num_col]

            # Plot the lineplot
            ax.plot(
                    times,
                    data,
                    marker='',
                    color='black',
                    linewidth=1.9,
                    alpha=0.9,
                    )

            # Same limits for everybody!
            ax.set_xlim(times[0], times[-1])
            ax.set_ylim(min_bold, max_bold)

            # Add title
            ax.set_title(
                    area,
                    loc='left',
                    fontsize=20,
                    fontweight=0,
                    color='black'
                    )

        fig.tight_layout()
        plt.savefig(os.path.join(
            self.plot_folder,
            'bold_overview_synaptic_input_current.png'
            ))
        plt.clf()
        plt.close(fig)
        plt.style.use('default')

    def plotAllFiringRatesSummary_cluster_resolved(self, t_start_=None, t_stop_=None):
        """
        Generates a summary plot giving an overview over the spiking activity
        in the network. Each subplot is the time course of the spike rates
        averaged across all neurons in a given area.
        """
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        rate_hist, _ = self.firingRateHistogram(group=['area', 'cluster'])
        rate_hist = rate_hist * 1000.  # Given in ms,
                                                        # transfrom to s
        resolution = self.ana_dict['rate_histogram_binsize']

        if self.ana_dict['plotAllFiringRatesSummary']['t_start']:
            t_start = self.ana_dict['plotAllFiringRatesSummary']['t_start'] \
                    * resolution
        elif t_start_:
            t_start = t_start_
        else:
            t_start = 0.
        t_start = int(t_start)

        if self.ana_dict['plotAllFiringRatesSummary']['t_stop']:
            t_stop = self.ana_dict['plotAllFiringRatesSummary']['t_stop'] \
                    * resolution
        elif t_stop_:
            t_stop = t_stop_
        else:
            t_stop = self.sim_dict['t_sim'] \
                    * resolution
        t_stop = int(t_stop)

        rate_hist = rate_hist.apply(lambda x: x[t_start:t_stop])

        times = np.arange(t_start, t_stop, resolution) / 1000.  # To seconds

        real_len = rate_hist.apply(len).iloc[0]
        if real_len < len(times):
            times = times[:real_len]

        # Determine max rate for y axis
        max_rate = rate_hist.apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllFiringRatesSummary']['num_col']
        num_row = math.ceil(len(rate_hist.index.levels[0]) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-darkgrid')

        for same_y_axis in [False, True]:
            # Iterate over unique clusters
            for cluster in rate_hist.index.levels[1]:
                fig, axes = plt.subplots(
                        num_row,
                        num_col,
                        figsize=(22, 22)
                        )

                # multiple line plot
                for num, area in enumerate(rate_hist.index.levels[0]):
                    data = rate_hist.loc[area, cluster]
                    # Find the right spot on the plot
                    ax = axes[num // num_col][num % num_col]

                    # Plot the lineplot
                    ax.plot(
                            times,
                            data,
                            marker='',
                            color='black',
                            linewidth=1.9,
                            alpha=0.9,
                            )

                    # Same limits for everybody!
                    ax.set_xlim(times[0], times[-1])
                    if same_y_axis:
                        ax.set_ylim(0, max_rate)
                    else:
                        ax.set_ylim(0, max(data))

                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Rate (Spikes / s)')

                    # Add title
                    ax.set_title(
                            area,
                            loc='left',
                            fontsize=20,
                            fontweight=0,
                            color='black'
                            )

                fig.tight_layout()
                plt.savefig(os.path.join(
                    self.plot_folder,
                    f'spiking_overview_same_y_axis_{same_y_axis}_cluster_{cluster}_{t_start}_{t_stop}.png'
                    ))
                plt.clf()
                plt.close(fig)
        plt.style.use('default')

    def plotAllFiringRatesSummary_cluster_resolved_area_sorted(self):
        """
        Generates a summary plot giving an overview over the spiking activity
        in the network. Each subplot is the time course of the spike rates
        averaged across all neurons in a given area.
        """
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        rate_hist, _ = self.firingRateHistogram(group=['area', 'cluster'])
        rate_hist = rate_hist * 1000.  # Given in ms,
                                                        # transfrom to s
        resolution = self.ana_dict['rate_histogram_binsize']

        if self.ana_dict['plotAllFiringRatesSummary']['t_start']:
            t_start = self.ana_dict['plotAllFiringRatesSummary']['t_start'] \
                    * resolution
        else:
            t_start = 0.
        t_start = int(t_start)

        if self.ana_dict['plotAllFiringRatesSummary']['t_stop']:
            t_stop = self.ana_dict['plotAllFiringRatesSummary']['t_stop'] \
                    * resolution
        else:
            t_stop = self.sim_dict['t_sim'] \
                    * resolution
        t_stop = int(t_stop)

        rate_hist = rate_hist.apply(lambda x: x[t_start:t_stop])

        times = np.arange(t_start, t_stop, resolution) / 1000.  # To seconds

        # Initialize the figure
        plt.style.use('seaborn-darkgrid')

        for same_y_axis in [False, True]:
            # Iterate over unique clusters
            # for cluster in rate_hist.index.levels[1]:
            for area in rate_hist.index.levels[0]:
                # Set some number for number of columns
                Q = self.network.params['cluster'][area]['Q']
                num_col = 5  # self.ana_dict['plotAllFiringRatesSummary']['num_col']
                num_row = math.ceil(Q / num_col)

                fig, axes = plt.subplots(
                        num_row,
                        num_col,
                        figsize=(22, 22)
                        )

                # Determine max rate for y axis
                max_rate = rate_hist.loc[area].apply(max).max()

                # multiple line plot
                # for num, area in enumerate(rate_hist.index.levels[0]):
                for num, cluster in enumerate(rate_hist.index.levels[1]):
                    data = rate_hist.loc[area, cluster]
                    # Find the right spot on the plot
                    if num_row > 1:
                        ax = axes[num // num_col][num % num_col]
                    else:
                        ax = axes[num % num_col]

                    # Plot the lineplot
                    ax.plot(
                            times,
                            data,
                            marker='',
                            color='black',
                            linewidth=1.9,
                            alpha=0.9,
                            )

                    # Same limits for everybody!
                    ax.set_xlim(times[0], times[-1])
                    if same_y_axis:
                        ax.set_ylim(0, max_rate)
                    else:
                        ax.set_ylim(0, max(data))

                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Rate (Spikes / s)')

                    # Add title
                    ax.set_title(
                            area,
                            loc='left',
                            fontsize=20,
                            fontweight=0,
                            color='black'
                            )

                fig.tight_layout()
                plt.savefig(os.path.join(
                    self.plot_folder,
                    f'spiking_overview_same_y_axis_{same_y_axis}_area_{area}.png'
                    ))
                plt.clf()
                plt.close(fig)
        plt.style.use('default')

    def plotAllFiringRatesSummary(self):
        """
        Generates a summary plot giving an overview over the spiking activity
        in the network. Each subplot is the time course of the spike rates
        averaged across all neurons in a given area.
        """
        if not hasattr(self, 'popGids'):
            self.popGids = self._readPopGids()
        if not hasattr(self, 'rate_hist_areas'):
            _, self.rate_hist_areas = self.firingRateHistogram()
        rate_hist_areas = self.rate_hist_areas * 1000.  # Given in ms,
                                                        # transfrom to s
        resolution = self.ana_dict['rate_histogram_binsize']

        if self.ana_dict['plotAllFiringRatesSummary']['t_start']:
            t_start = self.ana_dict['plotAllFiringRatesSummary']['t_start'] \
                    * resolution
        else:
            t_start = 0.
        t_start = int(t_start)

        if self.ana_dict['plotAllFiringRatesSummary']['t_stop']:
            t_stop = self.ana_dict['plotAllFiringRatesSummary']['t_stop'] \
                    * resolution
        else:
            t_stop = self.sim_dict['t_sim'] \
                    * resolution
        t_stop = int(t_stop)

        rate_hist_areas = rate_hist_areas.apply(lambda x: x[t_start:t_stop])

        times = np.arange(t_start, t_stop, resolution) / 1000.  # To seconds

        # Determine max rate for y axis
        max_rate = rate_hist_areas.apply(max).max()

        # Set some number for number of columns
        num_col = self.ana_dict['plotAllFiringRatesSummary']['num_col']
        num_row = math.ceil(len(rate_hist_areas) / num_col)

        # Initialize the figure
        plt.style.use('seaborn-darkgrid')

        for same_y_axis in [False, True]:
            fig, axes = plt.subplots(
                    num_row,
                    num_col,
                    figsize=(22, 22)
                    )

            # multiple line plot
            for num, (area, data) in enumerate(rate_hist_areas.items()):
                # Find the right spot on the plot
                ax = axes[num // num_col][num % num_col]

                # Plot the lineplot
                ax.plot(
                        times,
                        data,
                        marker='',
                        color='black',
                        linewidth=1.9,
                        alpha=0.9,
                        )

                # Same limits for everybody!
                ax.set_xlim(times[0], times[-1])
                if same_y_axis:
                    ax.set_ylim(0, max_rate)
                else:
                    ax.set_ylim(0, max(data))

                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Rate (Spikes / s)')

                # Add title
                ax.set_title(
                        area,
                        loc='left',
                        fontsize=20,
                        fontweight=0,
                        color='black'
                        )

            fig.tight_layout()
            plt.savefig(os.path.join(
                self.plot_folder,
                'spiking_overview_same_y_axis_{}.png'.format(same_y_axis)
                ))
            plt.clf()
            plt.close(fig)
        plt.style.use('default')

    @timeit
    def _readSpikes(self):
        """
        Reads SpikeTrains of the simulation.

        Returns
        -------
        spikes : Series of lists of SpikeTrains
        """
        try:
            spikes = pd.read_pickle(
                os.path.join(self.ana_folder, 'spikes.pkl')
            )
        except FileNotFoundError:
            print('Loading SpikeTrains from gdf')
            spikes = self._readSpikesFromGDF()
            # Save spikes to pickle for faster read access
            spikes.to_pickle(os.path.join(self.ana_folder, 'spikes.pkl'))
        return spikes

    @timeit
    def _readPopGids(self):
        """
        Reads the min / max GID for each population.

        Returns
        -------
        popGids : DataFrame
            Columns ['minGID', 'maxGID']
        """
        try:
            popGids = pd.read_pickle(
                os.path.join(self.ana_folder, 'population_GIDs.pkl')
            )
        except FileNotFoundError:
            rec_dir = os.path.join(self.simulation.data_dir, 'recordings')
            popGids = pd.read_csv(
                    os.path.join(rec_dir,
                        'network_gids.txt'),
                    names=['area', 'layer', 'pop', 'cluster', 'minGID', 'maxGID']
                    )
            popGids['layer'] = popGids['layer'].astype(int).astype(str)
            popGids['cluster'] = popGids['cluster'].astype(int).astype(str)
            popGids = popGids.set_index(['area', 'layer', 'pop', 'cluster'])
            popGids['pop_size'] = popGids['maxGID'] - popGids['minGID'] + 1
            # Save popGids to pickle for faster read access
            popGids.to_pickle(
                os.path.join(self.ana_folder, 'population_GIDs.pkl')
            )

        # Calculate recorded population size
        spike_fraction = self.simulation.params['recording_dict']['Nrec_spikes_fraction']
        popGids['recorded_pop_size'] = (popGids['maxGID'] - popGids['minGID'] + 1) * spike_fraction

        return popGids

    def _readSpikesFromGDF(self):
        """
        Reads spikes from gdf output files using pandas.
        Stores all SpikeTrains for one population in a list which in
        turn is contained in a Series.

        Returns
        -------
        spikes : Series of arrays of arrays containing spike timings.
        """
        if False:
            print(f"reading spikes from GDF using FIRST block")
            # Read in population gids
            popGids = self.popGids
            # Setup read_csv arguments
            csv_args = {'names': ['gid', 't'],
                        'sep': '\t',
                        'index_col': False}
            # Adapt csv arguments and file ending according to used NEST version
            if self.network.params['USING_NEST_3']:
                csv_args.update({'skiprows': 3})
                file_ending = '*.dat'
            else:
                file_ending = '*.gdf'
            # glob all spikes files
            gdf_files = glob.glob(os.path.join(self.sim_folder, 'recordings', file_ending))
            # Read in all gdf files into a big dataframe with two columns, gid and
            # t. Magically this seems to work in parallel
            ts = time.time()
            spikes = pd.concat(
                    (
                        pd.read_csv(
                            f,
                            **csv_args
                            ) for f in gdf_files
                    ),
                    ignore_index=True
                    )

            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'reading in spikes took {passed_time} s')
            ts = time.time()
            # Sort spikes, first gid, then t
            spikes = spikes.sort_values(['gid', 't'])
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'spike sorting took {passed_time} s')
            # Create a spiketrain (i.e. numpy arrays) for every gid and store them
            # in a cell
            ts = time.time()
            spikes = spikes.groupby('gid').apply(
                    lambda group: group.t.values
                    ).reset_index()
            spikes.columns = ['gid', 't']
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'spike train creation took {passed_time} s')

            no_spiking = []
            for row in popGids.itertuples():
                area = row.Index[0]
                layer = row.Index[1]
                pop = row.Index[2]
                cluster = row.Index[3]
                first_gid = row.minGID
                last_gid = row.maxGID
                try:
                    ts = time.time()
                    first = np.where(
                            (spikes.gid <= last_gid) & (spikes.gid >= first_gid)
                            )[-1][0]
                    last = np.where(
                            (spikes.gid <= last_gid) & (spikes.gid >= first_gid)
                            )[-1][-1]
                    spikes.loc[first:last, 'area'] = area
                    spikes.loc[first:last, 'layer'] = layer
                    spikes.loc[first:last, 'pop'] = pop
                    spikes.loc[first:last, 'cluster'] = cluster
                    te = time.time()
                    passed_time = round(te - ts, 3)
                    print(f'assigning correct {area} {layer} {pop} {cluster} naming to every gid took {passed_time} s')
                except IndexError:
                    no_spiking.append([area, layer, pop, cluster])
                    pass
            ts = time.time()
            spikes['layer'] = spikes['layer'].astype(int).astype(str)
            spikes['cluster'] = spikes['cluster'].astype(int).astype(str)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'converting all layer and cluster names to ints took {passed_time} s')
            ts = time.time()
            spikes = spikes.groupby(
                    ['area', 'layer', 'pop', 'cluster']
                    ).apply(lambda group: group.t.values)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'grouping by area layer pop cluster took {passed_time} s')

            ts = time.time()
            for (area, layer, pop, cluster) in no_spiking:
                spikes.loc[area, layer, pop, cluster] = np.array([])
            spikes = spikes.sort_index()
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'sorting by index and filling empty places took {passed_time} s')
            return spikes
        else:
            # This routine should be used when the simulated time is long, eg
            # 100 seconds. It is not faster when analyzing 10 seconds of
            # biological time. But when we simulate long and thus generate a
            # lot of data we need to do a lot of sorting. And pandas is bad at
            # this. In fact the original routine won't even finish sorting the
            # data for 100 seconds of biological time in 24 h. So for analyzing
            # larger datasets we need a more efficient spike reading in
            # routine. This routine relies a lot on GNU coreutils. E.g. sort
            # can be parallelized and it implements mergesort.
            # Note: this routine implicitly assumes that every area has spiked

            # ==============================================================================
            #                                   Definitions
            # ==============================================================================

            sorted_fn = 'combined_and_sorted_spiketrains.txt'
            available_cores = self.sim_dict['analysis']['local_num_threads']

            popGids = self.popGids.sort_values('minGID')

            # ==============================================================================
            #                                 Presorting data
            #
            # Here I presort all gdf files. Presorted files can easily be merged without
            # memory constraints. Also mergesort is quite fast.
            # The reason I use GNU sort instead of sorting in python is that GNU sort also
            # works in parallel. GNU sort uses 8 cores per default, which seems be a good
            # value as a rule of thumb. Furthermore I sort as many files as possible
            # simultaneously. I can launch available_cores / 8 jobs at a given time.
            # But: I haven't compared python sort vs GNU sort
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 15 minutes
            # ==============================================================================

            file_ending = '*.dat'
            gdf_files = glob.glob(os.path.join(self.rec_folder, file_ending))

            ts = time.time()
            pool = Pool(math.ceil(available_cores / 8))  # Gnu sort sorts in parallel with 8 threads
            _ = pool.map(shell_presort_all_dat, gdf_files)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'presorting data took {passed_time} s')

            # ==============================================================================
            #                                 mergesorting data
            #
            # In this step the data from before is mergesorted into a huge file. This file
            # contains all spikes, sorted by gid and time. GNU sort is used as it provides
            # a good mergesorting algorithm. This is way I used it instead of a pythonic
            # way. But: I haven't compared python sort vs GNU sort
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 22 minutes
            # The resulting file, for 100 seconds of time, is 120 GB
            # ==============================================================================

            ts = time.time()
            subprocess.check_output(
                    f'export LC_ALL=C; sort -k1,1n -k2,2n -m --parallel=8 {self.rec_folder}/*_sorted.txt > {self.rec_folder}/{sorted_fn}',
                    shell=True
                    )
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'mergesorting data took {passed_time} s')

            # ==============================================================================
            #            Splitting sorted data into population resolved data files
            #
            # Now we have this gigantic sorted datafile (120GB) which we would like to
            # split into population specific files. Meaning: Every text file contains all
            # spikes of the given population in a sorted manner. Splitting such a big file
            # takes quite some time. The algorithm works in the following way:
            # 1) Find a population in the middle of the popGids Dataframe
            # 2) Split according to this population
            # 3) Now we have two dataframes, one is left and the other one is right. These
            #    Dataframes also need to be splitted. As they are independent, they can be
            #    split in parallel
            # 4) As the algorithm progresses, the chunks become more numerous (more
            #    parallelization) and smaller (faster).
            # 5) When all has been split, we are done :)
            #
            # I haven't benchmarked the GNU splitting routine against a pyhtonic approach
            #
            # This step takes, for 100 seconds of bio time on 64 cores, 80 minutes
            # ==============================================================================

            all_tmp = [(popGids, sorted_fn, 0, self.rec_folder)]

            for iteration in range(math.ceil(math.log2(len(popGids)))):
                ts = time.time()
                cores = min(int(2**iteration), available_cores)
                pool = Pool(cores)
                tmp = pool.starmap(split_files, all_tmp)
                all_tmp = []
                if tmp:
                    for x in range(len(tmp)):
                        if tmp[x] and len(tmp[x]) > 0:
                            all_tmp.append((tmp[x][0], tmp[x][1], tmp[x][4]+1, self.rec_folder))
                            if len(tmp[x]) > 3:
                                all_tmp.append((tmp[x][2], tmp[x][3], tmp[x][4]+1, self.rec_folder))
                te = time.time()
                passed_time = round(te - ts, 3)
                print(f'Splitting iteration {iteration} took {passed_time} s')

            # ==============================================================================
            #  Rename files such that the filename contains exact information on population
            #
            # We rename all files such that the filename gives away which population we are
            # looking at.
            #
            # This step is fast, around 1 minute
            # ==============================================================================

            ts = time.time()
            file_ending = sorted_fn + '_*'
            dem_files = glob.glob(os.path.join(self.rec_folder, file_ending))
            for fn in dem_files:
                if os.stat(fn).st_size != 0:
                    with open(fn) as f:
                        a = int(f.readline().split()[0])
                        exact_population = popGids[(popGids.minGID <= a) & (popGids.maxGID >= a)].index.tolist()[0]

                        # Assert that first and last spike are from the same population and
                        # that every sorting so far has done the correct thing.
                        check = int(subprocess.check_output(['tail', '-1', fn]).split()[0])
                        check_population = popGids[(popGids.minGID <= check) & (popGids.maxGID >= check)].index.tolist()[0]
                        assert exact_population == check_population

                        new_name = os.path.join(
                                self.rec_folder,
                                '_'.join(exact_population) + '.spiketrains'
                                )
                        os.rename(fn, new_name)

            # If a population did not spike we now generate empty spike files for those
            for (a, l, p, c), _ in popGids.iterrows():
                fn = os.path.join(self.rec_folder, f'{a}_{l}_{p}_{c}.spiketrains')
                if not os.path.isfile(fn):
                    open(fn, 'a').close()
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'Assigning correct names to datafiles took {passed_time} s')

            # ==============================================================================
            #          Pull spikefiles together, have one spiketrain per line, remove gids
            #
            # Before the text files look like this:
            #
            # 7 5.5
            # 7 8.6
            # 7 9.2
            # 8 5.7
            # 8 6.5
            #
            # We now make sure that every line contains a spiketrain and remove the gid
            # number as it is not important anymore. The result looks like this:
            #
            # 5.5 8.6 9.2
            # 5.7 6.5
            #
            # This step takes, for 100 seconds of bio time on 8 cores, 20 minutes. on 64
            # cores probably 3 minutes.
            # ==============================================================================

            ts = time.time()
            file_ending = '*.spiketrains'
            dem_files = glob.glob(os.path.join(self.rec_folder, file_ending))
            pool = Pool(available_cores)
            _ = pool.map(shell_spiketrainify, dem_files)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'spiketrainify data took {passed_time} s')

            # ==============================================================================
            #                        Read files into pandas Dataframes
            #
            # Reading in 100 seconds of bio time takes 20 minutes. Saving takes 1.5 minutes.
            # Note: This probably can be optimized
            # ==============================================================================

            ts = time.time()
            file_ending = '*.cut'
            spikes_txt = glob.glob(os.path.join(self.rec_folder, file_ending))
            all_spikes = pd.Series(index=popGids.index, dtype=object)
            for spike_txt_file in spikes_txt:
                area, layer, pop, cluster, *_ = spike_txt_file.split('/')[-1].replace('.','_').split('_')
                tmp = []
                with open(spike_txt_file) as f:
                    lines=f.readlines()
                    if lines[0] != '\n':
                        for line in lines:
                            tmp.append(np.fromstring(line, dtype=float, sep=' '))
                all_spikes[(area, layer, pop, cluster)] = np.array(tmp, dtype=object)
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'putting data into Series took {passed_time} s')

            ts = time.time()
            all_spikes.to_pickle(os.path.join(self.ana_folder, 'spikes.pkl'))
            te = time.time()
            passed_time = round(te - ts, 3)
            print(f'saving Series took {passed_time} s')
            return all_spikes

    def getHash(self):
        """
        Creates a hash from analysis parameters.

        Returns
        -------
        hash : str
            Hash for the simulation
        """
        hash_ = dicthash.generate_hash_from_dict(self.ana_dict)
        return hash_

    @timeit
    def read_in_experimental_functional_connectivity(self):
        """
        Reads in the experimental functional connectivity.

        Returns
        -------
        exp_FC : DataFrame
                 A Pandas DataFrame containing the experimental functional
                 connectivity
        """
        # Load experimental functional connectivity
        func_conn_data = {}
        fc_fn = os.path.join(
                base_path,
                'figures',
                'Schmidt2018_dyn',
                'Fig8_exp_func_conn.csv'
                )

        with open(fc_fn, 'r') as f:
            myreader = csv.reader(f, delimiter='\t')
            # Skip first 3 lines
            next(myreader)
            next(myreader)
            next(myreader)
            areas = next(myreader)
            for line in myreader:
                dict_ = {}
                for i in range(len(line)):
                    dict_[areas[i]] = float(line[i])
                func_conn_data[areas[myreader.line_num - 5]] = dict_

        tmp = pd.DataFrame(func_conn_data)
        exp_FC = tmp.loc[complete_area_list, complete_area_list]
        return exp_FC

    @timeit
    def write_out_exp_and_simulated_cv(self):
        """
        This function calculates the cv_isi of V1 of the simulation and writes
        out this value as well as the experimental cv isi.
        """
        if not hasattr(self, 'exp_spikes'):
            self.exp_spikes = self.read_in_experimental_spikes()
        t_start = self.ana_dict['cv']['t_start']
        t_stop = self.ana_dict['cv']['t_stop']
        # Calculate cv isi from experiment
        exp_spikes_cv = cvIsi(self.exp_spikes)

        # Calculate area resolved cv isi from simulation
        # First groupby area
        # Second calculate it
        spks_tmp = self.spikes.groupby(['area']).apply(lambda x: [z for y in x.values for z in y])
        sim_cv_v1 = cvIsi(spks_tmp.loc['V1'], t_start=t_start, t_stop=t_stop)

        outfile = open(
                os.path.join(
                    self.ana_folder,
                    'cv_v1.txt'
                    ), 'w')

        outfile.write(
                ('exp_cv'
                f': {exp_spikes_cv}\n')
                )
        outfile.write(
                ('simulated_cv'
                f': {sim_cv_v1}')
                )

        outfile.close()

    @timeit
    def calculate_fc_correlations(self):
        """
        This function calculates the correlations between the simulated
        functional connectivity and the experimental data. The result is
        written to a text file.
        """
        if not hasattr(self, 'exp_FC'):
            self.exp_FC = self.read_in_experimental_functional_connectivity()
        if not hasattr(self, 'BOLD_correlation_syn'):
            self.BOLD_correlation_syn = self.calculateBOLDConnectivity(mode='bold_synaptic_input_current')
        if not hasattr(self, 'BOLD_correlation_rates'):
            self.BOLD_correlation_rates = self.calculateBOLDConnectivity(mode='rates')
        if not hasattr(self, 'fc_synaptic_currents'):
            self.fc_synaptic_currents = self.calculate_fc_based_on_synaptic_currents()

        exp_FC = self.exp_FC.sort_index(axis=0).sort_index(axis=1).values
        BOLD_correlation_syn = self.BOLD_correlation_syn.sort_index(axis=0).sort_index(axis=1).values
        BOLD_correlation_rates = self.BOLD_correlation_rates.sort_index(axis=0).sort_index(axis=1).values
        fc_synaptic_currents = self.fc_synaptic_currents.sort_index(axis=0).sort_index(axis=1).values

        np.fill_diagonal(exp_FC, np.NaN)
        np.fill_diagonal(BOLD_correlation_syn, np.NaN)
        np.fill_diagonal(BOLD_correlation_rates, np.NaN)
        np.fill_diagonal(fc_synaptic_currents, np.NaN)

        exp_FC = exp_FC.ravel()
        BOLD_correlation_syn = BOLD_correlation_syn.ravel()
        BOLD_correlation_rates = BOLD_correlation_rates.ravel()
        fc_synaptic_currents = fc_synaptic_currents.ravel()

        exp_FC = exp_FC[~np.isnan(exp_FC)]
        BOLD_correlation_syn = BOLD_correlation_syn[~np.isnan(BOLD_correlation_syn)]
        BOLD_correlation_rates = BOLD_correlation_rates[~np.isnan(BOLD_correlation_rates)]
        fc_synaptic_currents = fc_synaptic_currents[~np.isnan(fc_synaptic_currents)]

        exp_FC_BOLD_correlation_syn = np.corrcoef(exp_FC, BOLD_correlation_syn)[0, 1]
        exp_FC_BOLD_correlation_rates = np.corrcoef(exp_FC, BOLD_correlation_rates)[0, 1]
        exp_FC_fc_synaptic_currents = np.corrcoef(exp_FC, fc_synaptic_currents)[0, 1]
        BOLD_correlation_rates_BOLD_correlation_syn = np.corrcoef(BOLD_correlation_rates, BOLD_correlation_syn)[0, 1]
        BOLD_correlation_rates_fc_synaptic_currents = np.corrcoef(BOLD_correlation_rates, fc_synaptic_currents)[0, 1]
        BOLD_correlation_syn_fc_synaptic_currents = np.corrcoef(BOLD_correlation_syn, fc_synaptic_currents)[0, 1]

        outfile = open(
                os.path.join(
                    self.ana_folder,
                    'correlations.txt'
                    ), 'w')

        outfile.write(
                ('exp_FC_BOLD_correlation_syn'
                f': {exp_FC_BOLD_correlation_syn}\n')
                )
        outfile.write(
                ('exp_FC_BOLD_correlation_rates'
                f': {exp_FC_BOLD_correlation_rates}\n')
                )
        outfile.write(
                ('exp_FC_fc_synaptic_currents'
                f': {exp_FC_fc_synaptic_currents}\n')
                )
        outfile.write(
                ('BOLD_correlation_rates_BOLD_correlation_syn'
                f': {BOLD_correlation_rates_BOLD_correlation_syn}\n')
                )
        outfile.write(
                ('BOLD_correlation_rates_fc_synaptic_currents'
                f': {BOLD_correlation_rates_fc_synaptic_currents}\n')
                )
        outfile.write(
                ('BOLD_correlation_syn_fc_synaptic_currents'
                f': {BOLD_correlation_syn_fc_synaptic_currents}')
                )

        outfile.close()

    @timeit
    def read_in_experimental_spikes(self):
        """
        Reads in experimental spike data. Copied from
        figures/Schmidt2018_dyn/Fig6_comparison_exp_spiking_data.py

        Returns
        -------
        exp_spikes : np.array
        """

        exp_spikes = np.array([])
        chu2014_path = self.ana_dict['chu2014_path']
        if chu2014_path:
            """
            Load data
            """

            data_pvc5 = {'ids': [], 'times': []}
            id_neuron = 0
            id_to_channel = {}

            load_path = os.path.join(chu2014_path,
                                     'crcns-pvc5/rawSpikeTime')

            for fn in os.listdir(load_path):
                # each file contains several neurons
                temp = loadmat(os.path.join(load_path, fn))['cluster_class']
                channel = fn.split('_')[-1].split('.')[0][2:]
                ids = set(temp[:, 0])
                for id_temp in ids:
                    mask = temp[:, 0] == id_temp
                    data_pvc5['ids'] += [id_neuron]
                    data_pvc5['times'] += [temp[:, 1][mask]]
                    id_to_channel[id_neuron] = channel
                    id_neuron += 1

            """
            Take only neurons of the first 113 electrodes into account which are
            less than 1 mm apart
            """
            ind_mm = []
            for i, idx in enumerate(data_pvc5['ids']):
                if int(id_to_channel[idx]) <= 112:
                    ind_mm.append(i)

            exp_spikes = np.array(data_pvc5['times'])[ind_mm]
        return exp_spikes

    def dump(self, base_folder):
        """
        Exports the full analysis specification. Creates a subdirectory of
        base_folder from the analysis hash where it puts all files.

        Parameters
        ----------
        base_folder : string
            Path to base output folder
        """
        hash_ = self.getHash()
        out_folder = os.path.join(base_folder, hash_)
        try:
            os.mkdir(out_folder)
        except OSError:
            pass

        # output simple data as yaml
        fn = os.path.join(out_folder, 'ana.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.ana_dict, outfile, default_flow_style=False)

    def load_IPython_shell(self):
        from IPython import embed
        embed()


def cvIsi(sts, t_start=None, t_stop=None):
    """
    Calculates the cv isi for a subset of neurons for a list of spiketrains.
    The spiketrains contained in the list of spiketrains have to contain at
    least 2 spikes for calculating the interspike interval isi. If this
    condition is not met this function returns 0.

    Parameters
    ----------
    sts : list
        list of spiketrains
    t_start : float
        First timestep to take into account
    t_stop :
        Last timestep to take into account

    Returns
    -------
    cv : float
        cv_isi
    """
    # ensure that there are only spiketrains of len > 1. This is
    # for np.diff which needs to output at least one value in order
    # for np.std and np.mean to return something which is not
    # np.nan. Otherwise return 0.
    if t_start:
        sts = [st[st >= t_start] for st in sts]
    if t_stop:
        sts = [st[st <= t_stop] for st in sts]
    sts = [st for st in sts if len(st) > 1]
    if len(sts) > 0:
        isi = np.array([np.diff(x, 1) for x in sts])
        cv = np.array([np.std(x) / np.mean(x) for x in isi])
        cv_isi = np.mean(cv)
        return cv_isi
    return 0.


def calculate_lv(isi, t_ref):
    """
    Calculates the local variation lv for a interspike interval distribution.

    Parameters
    ----------
    isi : list
        list of interspike intervals isi of a single neuron

    Returns
    -------
    lv : float
        lv
    """
    # NOTE Elephant and mam use different functions. Elephant uses the normal
    # local variation whereas mam uses the revised local variation. LV depends
    # on firing rate fluctuations which are caused by the refractory period.
    # This can be compensated for by subtracting the refractoriness constant,
    # t_ref, from the ISIs.
    # Here we take the revised local variation.
    # Multi area model function
    val = np.sum(
            (1. - 4 * isi[:-1] * isi[1:] / (isi[:-1] + isi[1:]) ** 2) \
                    * (1 + 4 * t_ref / (isi[:-1] + isi[1:]))
                    ) * 3 / (isi.size - 1.)
    # Elephant function
    # val = 3. * np.mean(np.power(np.diff(isi) / (isi[:-1] + isi[1:]), 2))
    return val


def LV(sts, t_ref, t_start=None, t_stop=None):
    """
    Calculates the local variation lv for a list of spiketrains sts. First we
    filter for spiktrains of length > 2 because otherwise the calculation
    fails. In this case we return 0. At the end we divide by the number of
    spiketrains, opposed to dividing by the number of neurons in a population.
    This way we take only neurons that have actually spiked into account.

    Parameters
    ----------
    sts : list
        list of spiketrains
    t_start : float
        First timestep to take into account
    t_stop :
        Last timestep to take into account

    Returns
    -------
    lv : float
        sum of single lvs, needs to normalized (=divided by neuron numbers)
    """
    # ensure that there are only spiketrains of len > 2.
    # So every spiketrain st in sts has len(st) > 1.
    if t_start:
        sts = [st[st >= t_start] for st in sts]
    if t_stop:
        sts = [st[st <= t_stop] for st in sts]
    sts = [st for st in sts if len(st) > 2]
    if len(sts) > 0:
        isi = np.array([np.diff(x, 1) for x in sts])
        lvr = np.array([calculate_lv(x, t_ref) for x in isi])
        lvr_isi = np.mean(lvr)
        return lvr_isi
    return 0.


def calc_rates(sts, sim_dict, ana_dict):
    """
    Calculates the histogram of rates of a list of spiketrains sts.
    NOTE: The units of the returned rates are in spikes / ms

    Returns
    -------
    rate : ndarray
        array of binned rates
    """
    t_max = sim_dict['t_sim']
    t_min = 0.
    resolution = ana_dict['rate_histogram_binsize']
    num_bins = int((t_max - t_min) / resolution)
    if len(sts) > 0:
        # Gives same output as:
        # elstat.time_histogram(
        #     sts, binsize=resolution*pq.ms, output='counts'
        # )
        counts, _ = np.histogram(
                np.concatenate(sts).ravel(),
                bins=num_bins,
                # range=(t_min, t_max)  # or range=(t_min + resolution / 2., t_max + resolution / 2.) # I don't remember why I commented it out. TODO Need to look into this again
                )
        rate = counts * 1. / resolution
        return rate
    return np.zeros(num_bins)


def correlation(sts, ana_dict):
    """
    Calculates the correlation coefficients for a subset of neurons for a
    list of spiketrains.
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    ana_dict : dictionary
        dictionary containing values

    Returns
    -------
    cc : float
        Correlation coefficient
    """
    subsample = ana_dict['correlation_coefficient']['subsample']
    _, hist = instantaneous_spike_count(sts, ana_dict)
    rates = strip_binned_spiketrains(hist)[:subsample]
    # Need at least 2 spiketrains
    if len(rates) > 1:
        cc = np.corrcoef(rates)
        cc = np.extract(1-np.eye(cc[0].size), cc)
        cc[np.where(np.isnan(cc))] = 0.
        return np.mean(cc)
    return 0.


def instantaneous_spike_count(data, ana_dict):
    '''
    Create a histogram of spike trains.
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    ana_dict : dictionary
        dictionary containing values

    Returns
    -------
    bins : np.array
        Bins
    hist : np.array
        Histogram
    '''
    tbin = ana_dict['correlation_coefficient']['tbin']
    tmin = ana_dict['correlation_coefficient']['tmin']
    tmax = ana_dict['correlation_coefficient']['tmax']
    if tmin is None:
        tmin = np.min([np.min(x) for x in data if len(x) > 0])
    if tmax is None:
        tmax = np.max([np.max(x) for x in data if len(x) > 0])
    assert(tmin < tmax)
    bins = np.arange(tmin, tmax + tbin, tbin)
    hist = np.array([np.histogram(x, bins)[0] for x in data])
    return bins[:-1], hist


def strip_binned_spiketrains(sp):
    '''
    Removes binned spiketrains which do not contain a single spike
    Taken from correlation toolbox, available from
    https://github.com/INM-6/correlation-toolbox .

    Parameters
    ----------
    sp : np.array
        Array containing histogram.

    Returns
    -------
    sp_stripped : np.array
        Binned spiketrains with empty spiketrains removed.
    '''
    sp_stripped = np.array(
            [x for x in sp if abs(np.max(x) - np.min(x)) > 1e-16]
            )
    return sp_stripped

def E(f_in, rho):
    """
    fraction of oxygen extracted from the inflowing blood

    Buxton et al. 1998
    """
    tmp = 1 - np.power(1 - rho, 1/f_in)
    return tmp

def balloon_windkessel(t, w, z):
    """
    f_in : inflow from the venouscompartment
    v : Volume

    z : neuronal activity

    TVB implementations:
    https://github.com/the-virtual-brain/tvb-hpc/blob/master/tvb_hpc/bold.py
    https://github.com/the-virtual-brain/tvb-root/blob/bb3d3c91fb2ba20273b9a065943a141002b07229/scientific_library/tvb/analyzers/fmri_balloon.py
    neurolib implementation:
    https://github.com/neurolib-dev/neurolib/blob/27ea47aa33d83d080952954600380d33b6e38d34/neurolib/models/bold/timeIntegration.py
    WholeBrain implementation:
    https://github.com/dagush/WholeBrain/blob/faf3a557410c2f7a5942c0298c7ae332630d3eb5/functions/BOLDHemModel_Friston2003.py
    https://github.com/dagush/WholeBrain/blob/master/functions/BOLDHemModel_Stephan2008.py
    https://github.com/dagush/WholeBrain/blob/master/functions/BOLDHemModel_Stephan2007.py
    Deco lab
    https://github.com/decolab/cb-neuromod/blob/master/functions/BOLD.m

    Important papers:
    * K.J. Friston, L. Harrison, and W. Penny,
      Dynamic causal modelling, NeuroImage 19 (2003) 1273–1302
    * Klaas Enno Stephan, Nikolaus Weiskopf, Peter M. Drysdale, Peter A. Robinson, and Karl J. Friston
      Comparing hemodynamic models with DCM, NeuroImage 38 (2007) 387–401
   * K.J. Friston, Katrin H. Preller, Chris Mathys, Hayriye Cagnan, Jakob Heinzle, Adeel Razi, Peter Zeidman
     Dynamic causal modelling revisited, NeuroImage 199 (2019) 730–744
    """
    s, f_in, v, q = w
    # if f_in <= 0.:
    #     f_in = 1e-16

    eps = 1.  # Friston 2003; eps = 0.5  Friston 2000
    kappa = .65  # time-constant for signal decay or
                 # elimination [1/s], Friston 2003,
                 # kappa = 1 / tau_s, tau_s = 0.8 in Friston
                 # 2000
                 # For some reason some other code sources use
                 # this value reversed, is this a bug on their
                 # or our side? cf tvb, wholebrain, deco
    gamma = .41  # time-constant for autoregulatory feedback
                 # from blood flow [1/s], Friston 2003,
                 # gamma = 1 / tau_f
                 # tau_f = 0.4 in Friston 2000
                 # For some reason some other code sources use
                 # this value reversed, is this a bug on their
                 # or our side? cf tvb, wholebrain, deco
    tau = .98  # Mean transit time [s], time to traverse the
               # venous compartment
    alpha = .32  # Grubb's exponent, Grubb et al 1974
    rho = .34  # oxygen extraction fraction, sometimes called
               # E0

    f_out = np.power(v, 1 / alpha)  # Outflow

    # ========================================================
    #  For a description of these equations see Friston 2003
    # ========================================================

    # Change of vasodilatory signal s, dependent on neuronal
    # activity z, which is subject to autoregulatory feedback
    # Vasodilation is the widening of blood vessels
    dsdt = eps * z(t) - kappa * s - gamma * (f_in - 1)
    # assumption of dynamical system linking synaptic activity
    # and rCBF (regional cerbral blood flow) is linear
    dfdt = s
    # # Saturation constraint
    # if dfdt < 0. and f_in <= 0.:
    #     dfdt = 0.
    #     f_in = 1e-16
    # Rate of change of volume v
    dvdt = (f_in - f_out) / tau
    # change in deoxyhemoglobin q, delivery into venous
    # compartment minus expelled
    dqdt = (f_in * E(f_in, rho) / rho - f_out * q / v) / tau

    # # Saturation constraint
    # if dfdt < 0. and f_in <= 0.:
    #     dfdt = 1e-16
    #     f_in = 1e-16

    w_return = [dsdt, dfdt, dvdt, dqdt]
    return w_return

def calculate_BOLD(tmp):
    area = tmp[0]
    rate = tmp[1]
    # Assuming the rate is binned in [ms]
    t_vals = np.arange(0, len(rate)) / 1000.
    # Interpolate the rate in order to access values which do
    # not lie on the grid. The ode solver might access values
    # which are slightly out of the bounds of the interpolation
    # range, thus extrapolate those values.
    z_val = interp1d(
            t_vals,
            rate,
            bounds_error=False,
            fill_value='extrapolate'
            )

    s0 = 0.
    f0 = 1.
    v0 = 1.
    q0 = 1.

    w0 = [s0, f0, v0, q0]

    balloon_windkessel_solution = solve_ivp(
        lambda t, w: balloon_windkessel(t, w, z_val),
        (t_vals[0], t_vals[-1]),
        w0
        )

    _, _, v, q = balloon_windkessel_solution.y
    time = balloon_windkessel_solution.t

    rho = .34  # oxygen extraction fraction, sometimes called E0
    V0 = 0.02  # resting blood volume fraction

    # Buxton 1998
    k1 = 7 * rho
    k2 = 2.
    k3 = 2 * rho - 0.2

    BOLD = V0 * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
    return area, BOLD, time

def shell_presort_all_dat(fn):
    subprocess.check_output(
            f'export LC_ALL=C; f={fn}; tail -n +4 ${{f}} | sort -k1,1n -k2,2n --parallel=8 > ${{f%.dat}}_sorted.txt',
            shell=True
            )

def shell_spiketrainify(fn):
    lol2 = '''
    awk '
    {
      if($1==k)
      printf("%s"," ")
      else {
          if(NR!=1)
          print ""
        printf("%s\t",$1)
      }
      for(i=2;i<NF;i++)
        printf("%s ",$i)
      printf("%s",$NF)
      k=$1
    }
    END{
    print ""
    }' '''
    lol3 = f'{fn} > {fn}.sorted'
    # Use a literal tab character as the delimiter
    lol4 = f'''; cut -f2 {fn}.sorted > {fn}.sorted.cut'''
    lol5 = f'; rm {fn} {fn}.sorted'
    lol = lol2 + lol3 + lol4 + lol5

    subprocess.check_output(
            lol,
            shell=True
            )

def split_files(df, fn, iteration, rec_folder):
    if len(df) > 1:
        where_to_split = math.floor(len(df)/2)
        df_left = df.iloc[:where_to_split]
        df_right = df.iloc[where_to_split:]
        maxGID = df_left.iloc[-1].maxGID

        iteration_ = str(iteration)
        fn_left = fn + f'_left_{iteration_}'
        fn_right = fn + f'_right_{iteration_}'
        try:
            subprocess.check_output(
                    f'''a=$(awk '$1>{maxGID}{{print NR, $0; exit}}' {rec_folder}/{fn} | cut -d ' ' -f1); csplit -sf {rec_folder}/part.{fn}. {rec_folder}/{fn} $a; mv {rec_folder}/part.{fn}.00 {rec_folder}/{fn_left}; mv {rec_folder}/part.{fn}.01 {rec_folder}/{fn_right}''',
                    shell=True
                    )
            if fn != 'all_sorted_spiketrains2.txt':
                subprocess.check_output(f'rm {rec_folder}/{fn}', shell=True)
            return df_left, fn_left, df_right, fn_right, iteration
        except subprocess.CalledProcessError as e:
            print(f'Error during splitting file {fn}: {e}')
            return None

# Write out
# fn_in = '../../../mattention_bold/271e33bafd8104ea9e2919398cf26ed6/eafe8d385d61a3ab2fde17ee1cbea3ad/input_current.pkl'
# fn_out = '../../../mattention_bold/271e33bafd8104ea9e2919398cf26ed6/eafe8d385d61a3ab2fde17ee1cbea3ad/curr_in_271.npy'
# names_out =  '../../../mattention_bold/271e33bafd8104ea9e2919398cf26ed6/eafe8d385d61a3ab2fde17ee1cbea3ad/names_271.txt' 
#
# curr_in = pd.read_pickle(fn_in)
# curr_in.to_numpy().dump(fn_out)
# with open(names_out, 'w') as fn:
#         for area in curr_in.index:
#                     fn.write(area+'\n')

# Read in
# fn_curr = 'curr_in.npy'
# fn_names = 'names.txt'
#
# names = []
# with open(fn_names, 'r') as fn:
#     for line in fn.readlines():
#         names.append(line.split('\n')[0])
#
# curr_in = np.load(fn_curr)
# curr_in = pd.Series(curr_in, index=names)
