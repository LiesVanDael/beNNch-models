import os
import sys
import json
import nest

from multiarea_model import MultiAreaModel

"""
Create parameters.
"""

data_path = sys.argv[1]
label = sys.argv[2]

print("Load simulation parameters\n")

# Load simulation parameters
fn = os.path.join(data_path,
                  label,
                  '_'.join(('custom_params',
                            label,
                           str(nest.Rank()))))

try:
    with open(fn, 'r') as f:
        custom_params = json.load(f)
except FileNotFoundError:
    fn_base = os.path.join(data_path,
                      label,
                      '_'.join(('custom_params',
                                label
                               ))
                      )
    shutil.copy(fn_base, fn)
    with open(fn, 'r') as f:
        custom_params = json.load(f)

os.remove(fn)

print("Create network\n")
M = MultiAreaModel('benchmark',
                   simulation=True,
                   sim_spec=custom_params['sim_params']
                   )

print("Simulate\n")
M.simulation.simulate()
