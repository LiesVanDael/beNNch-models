import json
import os
import shutil

from config import base_path
from multiarea_model.default_params import nested_update, sim_params
try:
    from multiarea_model.sumatra_helpers import register_record
    sumatra_found = True
except ImportError:
    sumatra_found = False


def start_job(label, data_path, sumatra=False, reason=None, tag=None):
    """
    Create parameter folders.

    Parameters
    ----------

    label : str
        Simulation label identifying the simulation to be run.
        The function loads all necessary files from the subfolder
        identified by the label.
    data_path : str
        Path to folder in which parameters are saved.
  """
    # When we make a local assignment of sim_params (i.e. in the scope of this
    # function) the globally imported sim_params is shadowed. The global
    # statement tells Python that inside start_job sim_params refers to the
    # global variable sim_params, even if it's assigned in this function.
    global sim_params

    # Copy run_simulation script to simulation folder
    shutil.copy2(os.path.join(base_path, 'run_simulation.py'),
                 os.path.join(data_path, label))

    # Load simulation parameters
    fn = os.path.join(data_path,
                      label,
                      '_'.join(('custom_params',
                                label)))
    with open(fn, 'r') as f:
        custom_params = json.load(f)
    sim_params = nested_update(sim_params, custom_params['sim_params'])

    # Copy custom param file for each MPI process
    for i in range(sim_params['num_processes']):
        shutil.copy(fn, '_'.join((fn, str(i))))

    # If chosen, register simulation to sumatra
    if sumatra:
        if sumatra_found:
            register_record(label, reason=reason, tag=tag)
        else:
            raise ImportWarning('Sumatra is not installed, so'
                                'cannot register simulation record.')
