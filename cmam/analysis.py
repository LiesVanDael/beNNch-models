"""
This script is used to run an analysis from the given command-line
arguments:
1. Label of the simulation
2. Label of the network to be simulated

It initializes the network class and then runs the simulate method of
the simulation class instance.

This script should be used in the `jobscript_template` defined in the
config.py file. See config_template.py.
"""

import json
import nest
import os
import sys
import numpy as np

from config import data_path
from multiarea_model import MultiAreaModel
from multiarea_model.default_params import complete_area_list

label = sys.argv[1]
network_label = sys.argv[2]
fn = os.path.join(data_path,
                  label,
                  '_'.join(('custom_params',
                            label)))
with open(fn, 'r') as f:
    custom_params = json.load(f)

M = MultiAreaModel(network_label,
                   simulation=True,
                   analysis=True,
                   sim_spec=custom_params['sim_params'])

# M.analysis.fullAnalysis()

t_sim = M.analysis.sim_dict['t_sim']
for area in complete_area_list:
    for start_time in np.arange(0, t_sim, 1000):
        print(f'Plotting {area}')
        M.analysis.plotRasterArea(area, begin=start_time, end=start_time+1500.)
