import json, os, yaml, subprocess
from pydmc.core.mag import MagTools

def read_json(fjson):
    """
    Args:
        fjson (str) - file name of json to read
    
    Returns:
        dictionary stored in fjson
    """
    with open(fjson) as f:
        return json.load(f)

def write_json(d, fjson):
    """
    Args:
        d (dict) - dictionary to write
        fjson (str) - file name of json to write
    
    Returns:
        written dictionary
    """        
    with open(fjson, 'w') as f:
        json.dump(d, f)
    return d  

def read_yaml(fyaml):
    """
    Args:
        fyaml (str) - file name of yaml to read
    
    Returns:
        dictionary stored in fjson
    """
    with open(fyaml) as f:
        return yaml.safe_load(f)
    
def write_yaml(d, fyaml):
    """
    Args:
        d (dict) - dictionary to write
        fyaml (str) - file name of yaml to write
    
    Returns:
        written dictionary
    """        
    with open(fyaml, 'w') as f:
        yaml.dump(d, f)
    return d

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def is_slurm_job_in_queue(job_name, user_name='cbartel', fqueue='q.o'):
    with open(fqueue, 'w') as f:
        subprocess.call(['squeue','-u', user_name, '--name=%s' % job_name], stdout=f)
    names_in_q = []
    with open(fqueue) as f:
        for line in f:
            if 'PARTITION' not in line:
                names_in_q.append([v for v in line.split(' ') if len(v) > 0][2])
    if len(names_in_q) == 0:
        return False
    else:
        return True
    
def make_project_tree():
    
    project_dir = os.getcwd()
    tree = {'dev' : ['scripts', 'data', 'figures'],
            'results' : ['scripts', 'data', 'figures'],
            'prep' : ['notes', 'background'],
            'products' : ['drafts', 'slides', 'other']}
    for first_layer in tree:
        next_layers = tree[first_layer]
        for n in next_layers:
            that_layer = os.path.join(project_dir, first_layer, n)
            if not os.path.exists(that_layer):
                os.makedirs(that_layer)
            freadme = os.path.join(that_layer, 'README.md')
            if not os.path.exists(freadme):
                with open(freadme, 'w') as f:
                    f.write('This is a placeholder for the %s directory' % that_layer)
                    
def make_sub_for_launcher():
    flauncher_sub = os.path.join(os.getcwd(), 'sub_launch.sh')
    launch_job_name = '-'.join([os.getcwd().split('/')[-2], 'launcher'])
    with open(flauncher_sub, 'w') as f:
        f.write('#!/bin/bash -l\n')
        f.write('#SBATCH --nodes=1\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --time=4:00:00\n')
        f.write('#SBATCH --error=log_launch.e\n')
        f.write('#SBATCH --output=log_launch.o\n')
        f.write('#SBATCH --account=cbartel\n')
        f.write('#SBATCH --job-name=%s\n' % launch_job_name)
        f.write('#SBATCH --partition=msismall\n')
        f.write('\npython launch.py\n')
        
def is_calc_valid(structure,
                  standard,
                  xc,
                  calc,
                  mag,
                  magmom,
                  mag_override):
    """
    Returns:
        True if calculation should be launched; 
        False if some logic is violated
    """
    
    if standard == 'mp':
        if xc != 'ggau':
            return False
        
    if not mag_override:
        if mag == 'nm':
            if MagTools(structure).could_be_magnetic:
                return False
        elif 'afm' in mag:
            if not MagTools(structure).could_be_afm:
                return False
        elif mag == 'fm':
            if not MagTools(structure).could_be_magnetic:
                return False
        
    if 'afm' in mag:
        if not magmom:
            return False
        
    if xc == 'metagga':
        if calc == 'loose':
            return False
        
    return True