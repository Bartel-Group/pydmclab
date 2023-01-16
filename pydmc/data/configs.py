import yaml, os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data")


def load_vasp_configs():
    with open(os.path.join(DATA_PATH, "_vasp_configs.yaml")) as f:
        return yaml.safe_load(f)


def load_launch_configs():
    with open(os.path.join(DATA_PATH, "_launch_configs.yaml")) as f:
        return yaml.safe_load(f)


def load_slurm_configs():
    with open(os.path.join(DATA_PATH, "_slurm_configs.yaml")) as f:
        return yaml.safe_load(f)


def load_sub_configs():
    with open(os.path.join(DATA_PATH, "_sub_configs.yaml")) as f:
        return yaml.safe_load(f)


def load_partition_configs():
    with open(os.path.join(DATA_PATH, "_partition_configs.yaml")) as f:
        return yaml.safe_load(f)


def load_batch_vasp_analysis_configs():
    with open(os.path.join(DATA_PATH, "_batch_vasp_analysis_configs.yaml")) as f:
        return yaml.safe_load(f)
