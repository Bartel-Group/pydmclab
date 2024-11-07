import os
from shutil import copyfile

# The home path is needed to find the your pydmclab repository
#   pydmclab is assumed to be in /users/{number}/{username}/bin/pydmclab
#   and $HOME points to /users/{number}/{username}
HOME_PATH = os.environ["HOME"]

# where does the run.py script live?
CENTRAL_DIR = os.getcwd()

# what tree structure do we want to make?
#   we only specify the scripts directories here, as the rest of the tree is made in the launcher scripts
# This typically shouldn't change for the standard slab calcs
#   if you change the tree structure, you'll need to update the copy locations of the template launchers
TREE = {
    "bulk-templates": ["scripts"],
    "slabs": ["scripts"],
    "bulk-references": ["scripts"],
}

# check if the project tree already exists
# if it doesn't, assume this is the first time running the script and make the project tree/copy the launchers
FIRST_RUN = False
for k, v in TREE.items():
    for sub_dir in v:
        if not os.path.exists(os.path.join(CENTRAL_DIR, k, sub_dir)):
            FIRST_RUN = True
            break
    if FIRST_RUN:
        break

# minimum info required to start the template calcs from a query
# Note: This is not a complete list of possible inputs, just the minimum necessary info
#       you should check the template launcher prior to launching the template calcs
API_KEY = None
COMPOSITION = None

# should be generated automatically from the template results and passed to the slab/reference calcs
TEMPLATE_CIF_FILE = None
STRUC_ID = None
OVERRIDE_MAG = False

# minimum info required to start the slab calcs from a ground state template
# Note: This is not a complete list of possible inputs, just the minimum necessary info
#       you should check the slab launcher prior to launching the slab calcs
MILLER_INDICES = None


def make_project_tree(parent_dir=CENTRAL_DIR, project_tree=TREE):
    """
    Args:
        parent_dir (str) - directory to make project tree in
        project_tree (dict) - dictionary of directories to make

    Returns:
        None
    """
    for k, v in project_tree.items():
        os.makedirs(os.path.join(parent_dir, k), exist_ok=True)
        for sub_dir in v:
            os.makedirs(os.path.join(parent_dir, k, sub_dir), exist_ok=True)
    return None


def copy_templates_launcher():
    """
    Copies the template launcher to the correct scripts location and replaces the minimum required inputs
    """

    # copy the template launcher
    copyfile(
        f"{HOME_PATH}/bin/pydmclab/hpc_workflows/slab_calcs/demo_bulk-templates_launcher.py",
        os.path.join(CENTRAL_DIR, "bulk-templates", "scripts", "template-launcher.py"),
    )

    # read the template launcher
    with open(
        os.path.join(CENTRAL_DIR, "bulk-templates", "scripts", "template-launcher.py"),
        "r",
    ) as f:
        content = f.read()

    # replace the API_KEY and COMPOSITION
    content = content.replace("{{API_KEY}}", API_KEY)
    content = content.replace("{{COMPOSITION}}", COMPOSITION)

    # write the updated template launcher
    with open(
        os.path.join(CENTRAL_DIR, "bulk-templates", "scripts", "template-launcher.py"),
        "w",
    ) as f:
        f.write(content)

    return None


def copy_slabs_launcher():
    """
    Copies the slab launcher to the correct scripts location and replaces the minimum required inputs
    """

    # copy the slab launcher
    copyfile(
        f"{HOME_PATH}/bin/pydmclab/hpc_workflows/slab_calcs/demo_slabs-launcher.py",
        os.path.join(CENTRAL_DIR, "slabs", "scripts", "slabs-launcher.py"),
    )

    # read the slab launcher
    with open(
        os.path.join(CENTRAL_DIR, "slabs", "scripts", "slabs-launcher.py"), "r"
    ) as f:
        content = f.read()

    # replace the TEMPLATE_CIF_FILE, STRUC_ID, OVERRIDE_MAG, and MILLER_INDICES
    content = content.replace("{{TEMPLATE_CIF_FILE}}", TEMPLATE_CIF_FILE)
    content = content.replace("{{STRUC_ID}}", STRUC_ID)
    content = content.replace("{{OVERRIDE_MAG}}", str(OVERRIDE_MAG))
    content = content.replace("{{MILLER_INDICES}}", MILLER_INDICES)

    # write the updated slab launcher
    with open(
        os.path.join(CENTRAL_DIR, "slabs", "scripts", "slabs-launcher.py"), "w"
    ) as f:
        f.write(content)

    return None


def copy_references_launcher():
    """
    Copies the reference launcher to the correct scripts location and replaces the minimum required inputs
    """

    # copy the reference launcher
    copyfile(
        f"{HOME_PATH}/bin/pydmclab/hpc_workflows/slab_calcs/demo_bulk-references-launcher.py",
        os.path.join(
            CENTRAL_DIR, "bulk-references", "scripts", "references-launcher.py"
        ),
    )

    # read the reference launcher
    with open(
        os.path.join(
            CENTRAL_DIR, "bulk-references", "scripts", "references-launcher.py"
        ),
        "r",
    ) as f:
        content = f.read()

    # replace OVERRIDE_MAG
    content = content.replace("{{OVERRIDE_MAG}}", str(OVERRIDE_MAG))

    # write the updated reference launcher
    with open(
        os.path.join(
            CENTRAL_DIR, "bulk-references", "scripts", "references-launcher.py"
        ),
        "w",
    ) as f:
        f.write(content)

    return None


def main():

    if FIRST_RUN:
        make_project_tree()
        copy_templates_launcher()
        copy_slabs_launcher()
        copy_references_launcher()

    return None


if __name__ == "__main__":
    main()
