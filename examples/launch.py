from pydmc.VASPTools import VASPSetUp, VASPAnalysis
from pydmc.SubmitTools import SubmitTools
from pydmc.CompTools import CompTools
from pydmc.handy import read_json, write_json, read_yaml, write_yaml, is_slurm_job_in_queue
from pydmc.MagTools import MagTools
from pydmc.StrucTools import StrucTools
from pydmc.MPQuery import MPQuery

from pymatgen.core.structure import Structure

import os
import subprocess

# where is launch.py
SCRIPTS_DIR = os.getcwd()

# where to put .json files
DATA_DIR = SCRIPTS_DIR.replace('scripts', 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

# where to run calculations
CALCS_DIR = SCRIPTS_DIR.replace('scripts', 'calcs')
if not os.path.exists(CALCS_DIR):
    os.mkdir(CALCS_DIR)

# Chris' API key (replace with your own)
API_KEY = '***REMOVED***'

# how many AFM orderings I want to sample
MAX_AFM_IDX = 2

# which magnetic configurations I want to calculate (note: AFM "mag" will be "afm_0", "afm_1", ... "afm_N"
MAGS = ['nm', 'fm'] + ['afm_%s' % str(int(v)) for v in range(MAX_AFM_IDX)] if MAX_AFM_IDX != 0 else []

# what kind of calculation I want to do
CALC = 'relax'

# what kind of DFT I want to do
XC = 'metagga'

# what "standard" settings I want to use
STANDARD = 'dmc'

# which MP IDs I want to calculate
MPIDS = ['mp-22584', 'mp-770495']

def query_mp(remake=False):
    """
    get starting structures from MP
    """
    fjson = os.path.join(DATA_DIR, 'query.json')
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)

    mpq = MPQuery(API_KEY)
    IDs = ['mp-22584', 'mp-770495']

    d = {}
    for mpid in IDs:
        s = mpq.get_structure_by_material_id(mpid)
        formula = StrucTools(s).compact_formula
        d[mpid] = {'structure' : s.as_dict()}
        
    return write_json(d, fjson)

def get_afm_magmoms(query, remake=False):
    """
    get MAGMOMs for AFM calculations
    """

    fjson = os.path.join(DATA_DIR, 'magmoms.json')
    if os.path.exists(fjson) and not remake:
        return read_json(fjson)
        
    magmoms = {}
    for mpid in query:
        magmoms[mpid] = {}
        s = Structure.from_dict(query[mpid]['structure'])
        magtools = MagTools(s)
        afm_strucs = magtools.get_antiferromagnetic_structures
        for i in range(len(afm_strucs)):
            magmoms[mpid][i] = afm_strucs[i].site_properties['magmom']
    
    return write_json(magmoms, fjson)

def get_job_name(launch_dir):
    """
    descriptive job name
    """
    return '-'.join(launch_dir.split('/')[-2:])

def get_launch_dir(mpid, mag):
    """
    where I want to "$ sbatch sub.sh" to launch all calcs in a tree
    """

    launch_dir = os.path.join(CALCS_DIR, mpid, mag)
    if not os.path.exists(launch_dir):
        os.makedirs(launch_dir)
    return launch_dir

def prepare_calc_and_launch(mpid, 
                            mag,
                            ready_to_launch=False,
                            fresh_restart=False,
                            refresh_configs_yaml=False,
                            user_name='cbartel'):
    """
    prepare directories for launch
    make subdirectories for all calculations for a given mpid, mag
    populate with VASP input files
    populate with submission scripts
    submits calculation to the queue (if ready_to_launch = True)
    
    Args:
    
    mpid (str) - Materials Project ID
    mag (str) - 'nm' = nonmagnetic, 'fm' = ferromagnetic, 'afm_#' = antiferromagnetic with # idx ordering
    ready_to_launch (bool) - True if submit jobs now; False if you want to check directories first
    fresh_restart (bool) - if True, delete progress and start over; if False, pick up where you left off
    refresh_configs_yaml (bool) - if True, grab the base_configs.yaml from pydmc; if False, keep the one you've been editing in SCRIPTS_DIR
    user_name (str) - change to your name *******DON'T FORGET THIS ONE ********
    
    """
    # if you want to start from pydmc yaml in SCRIPTS_DIR
    if refresh_configs_yaml:
        fyaml = os.path.join(os.getcwd(), 'base_configs.yaml')
        if os.path.exists(fyaml):
            os.remove(fyaml)

    # load dictionaries from DATA_DIR/*json
    query = query_mp()
    magmoms = get_afm_magmoms(query)

    # determine launch directory
    launch_dir = get_launch_dir(mpid, mag)

    # determine magmom if calc is afm
    if 'afm' in mag:
        afm_idx = mag.split('_')[1]
        if str(afm_idx) in magmoms[mpid]:
            magmom = magmoms[mpid][str(afm_idx)]
        else:
            # skip if afm_* not in magmoms (eg we didn't make that many magmoms)
            return
    else:
        # set magmom to None if not AFM (handled elsewhere in VASPSetUp)
        magmom = None 

    
    # put POSCAR in launch_dir if it's not there already
    struc = Structure.from_dict(query[mpid]['structure'])
    launch_pos = os.path.join(launch_dir, 'POSCAR')
    if not os.path.exists(launch_pos) or fresh_restart:
        struc.to(filename=launch_pos, fmt='POSCAR')

    # check to see if job is in the queue (**** MAKE SURE YOUR USERNAME IS AN ARG ****)
    job_name = get_job_name(launch_dir)
    if is_slurm_job_in_queue(job_name, user_name=user_name):
        print('%s already in queue, not messing with it' % job_name)
        return

    # specify configurations that you want to change in base_configs.yaml (some may already be there - that's OK)
    user_configs = {}
    user_configs['mag'] = mag # use the mag I'm iterating through
    user_configs['fresh_restart'] = fresh_restart # restart or not according to arg
    user_configs['standard'] = STANDARD # use the standard we decided on above
    user_configs['calc'] = CALC # use the calc we decided on above
    user_configs['xc'] = XC # use the exchange-correlation we decided above
    user_configs['job-name'] = job_name # use a unique job name
    user_configs['partition'] = 'msismall' # choose a partition (change this if jobs aren't starting quickly)
    user_configs['nodes'] = 1 # choose # nodes (increase if jobs are taking too long)
    user_configs['ntasks'] = 16 # choose # tasks (increase if jobs are taking too long; decrease if trouble getting jobs running in a "small" queue


    sub = SubmitTools(launch_dir=launch_dir,
                      magmom=magmom,
                      user_configs=user_configs)
    sub.write_sub
    if ready_to_launch:
        os.chdir(launch_dir)
        subprocess.call(['sbatch', 'sub.sh'])
        os.chdir(SCRIPTS_DIR)

def prepare_calcs(query, magmoms, fresh_restart=False, refresh_configs_yaml=False, user_name='cbartel'):
    """
    
    """
    if refresh_configs_yaml:
        fyaml = os.path.join(os.getcwd(), 'base_configs.yaml')
        if os.path.exists(fyaml):
            os.remove(fyaml)
    for standard in ['mp', 'dmc']:
        for mpid in query:
            for mag in MAGS:
                if 'afm' in mag:
                    afm_idx = mag.split('_')[1]
                    if str(afm_idx) in magmoms[mpid]:
                        magmom = magmoms[mpid][str(afm_idx)]
                    else:
                        continue
                launch_dir = get_launch_dir(mpid, mag, standard)
                print('\n\n\n working on %s' % launch_dir)

                magmom = None if 'a' not in mag else magmom
                struc = Structure.from_dict(query[mpid]['structure'])
                launch_pos = os.path.join(launch_dir, 'POSCAR')
                if not os.path.exists(launch_pos) or fresh_restart:
                    struc.to(filename=launch_pos, fmt='POSCAR')
                job_name = get_jobname(launch_dir)
                if is_slurm_job_in_queue(job_name, user_name=user_name):
                    print('%s already in queue, not messing with it' % job_name)
                    continue
                sub = SubmitTools(launch_dir=launch_dir,
                                  magmom=magmom,
                                  user_configs={'fresh_restart' : fresh_restart,
                                                mag : mag,
                                                'standard' : standard,
                                                'calc' : CALC,
                                                'xc' : XC,
                                                'job-name' : get_jobname(launch_dir),
                                                'partition' : 'msigpu',
                                                'machine' : 'msi',
                                                'nodes' : 1,
                                                'ntasks' : 8})
                sub.write_sub
                os.chdir(launch_dir)
                subprocess.call(['sbatch', 'sub.sh'])
                os.chdir(SCRIPTS_DIR)

def main():
    query = query_mp()
    magmoms = get_afm_magmoms(query)

    ready_to_launch = False
    fresh_restart = False
    refresh_configs_yaml = False
    
    user_name = 'cbartel'

    for MPID in MPIDS:
        for MAG in MAGS:
            prepare_calc_and_launch(mpid=MPID,
                                    mag=MAG,
                                    ready_to_launch=ready_to_launch,
                                    fresh_restart=fresh_restart,
                                    refresh_configs_yaml=refresh_configs_yaml,
                                    user_name=user_name)


    return query, magmoms

if __name__ == '__main__':
    query, magmoms = main()
