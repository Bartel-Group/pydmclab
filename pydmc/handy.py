import json, os, yaml

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