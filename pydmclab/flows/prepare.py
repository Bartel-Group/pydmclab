from atomate2.vasp.jobs.base import BaseVaspMaker

from fireworks import LaunchPad

from jobflow import run_locally
from jobflow.managers.fireworks import flow_to_workflow

# NOTE: need to import makers of interest


def add_metadata_to_flow(flow, additional_fields, class_filter=BaseVaspMaker):
    """
    Return the flow with additional field(metadata) to the task doc.

    This allows adding metadata to the task-docs, could be useful
    to query results from DB.

    Parameters
    ----------
    flow:
    additional_fields : dict
        A dict with metadata.
    class_filter: .Maker
        The Maker to which additional metadata needs to be added

    Returns
    -------
    Flow
        Flow with added metadata to the task-doc.
    """
    flow.update_maker_kwargs(
        {
            "_set": {
                f"task_document_kwargs->additional_fields->{field}": value
                for field, value in additional_fields.items()
            }
        },
        dict_mod=True,
        class_filter=class_filter,
    )

    return flow


def add_pydmc_labels_to_flow(
    flow,
    class_filter=BaseVaspMaker,
    formula=None,
    ID=None,
    standard=None,
    mag=None,
    xc=None,
    calc=None,
):
    additional_fields = {
        "formula": formula,
        "ID": ID,
        "standard": standard,
        "mag": mag,
        "xc_calc": f"{xc}-{calc}",
    }
    flow = add_metadata_to_flow(flow, additional_fields, class_filter)
    return flow


def prepare_sh(submission_script):
    # NOTE: should just do this in the atomate2 launcher
    unload = [
        "mkl",
        "intel/2018.release",
        "intel/2018/release",
        "impi/2018/release_singlethread",
        "mkl/2018.release",
        "impi/intel",
    ]
    load = ["mkl/2021/release", "intel/cluster/2021"]
    with open(submission_script, "w") as f:
        for module in unload:
            f.write("module unload %s\n" % module)
        for module in load:
            f.write("module load %s\n" % module)
        f.write("ulimit -s unlimited\n")
        f.write("conda activate atomate2\n")


def prepare_workflow(Maker, structure, key, class_filter=BaseVaspMaker):
    formula, ID, standard, mag, xc_calc = key.split("--")
    xc, calc = xc_calc.split("-")
    flow = Maker.make(structure)
    flow = add_pydmc_labels_to_flow(
        flow,
        class_filter,
        formula=formula,
        ID=ID,
        standard=standard,
        mag=mag,
        xc=xc,
        calc=calc,
    )

    run_locally(flow, create_folders=True)


def prepare_firework(Maker, structure, key, class_filter=BaseVaspMaker):
    formula, ID, standard, mag, xc_calc = key.split("--")
    xc, calc = xc_calc.split("-")
    flow = Maker.make(structure)
    flow = add_pydmc_labels_to_flow(
        flow,
        class_filter,
        formula=formula,
        ID=ID,
        standard=standard,
        mag=mag,
        xc=xc,
        calc=calc,
    )

    # convert the flow to a fireworks WorkFlow object
    wf = flow_to_workflow(bandstructure_flow)

    # submit the workflow to the FireWorks launchpad
    lpad = LaunchPad.auto_load()
    lpad.add_wf(wf)
