import ase.io


def clean_md_log_and_traj_files(logfile: str, trajfile: str) -> None:
    """
    Removes any runs where no time has passed in the MD log and trajectory files.

    Args:
        logfile (str): Path to MD log file.
        trajfile (str): Path to MD trajectory file.

    Returns:
        None
    """

    lines_to_remove = []
    atoms_to_remove = []

    with open(logfile, "r", encoding="utf-8") as logf:
        lines = logf.readlines()
        for i, line in enumerate(lines):
            if i + 2 < len(lines) and "Time" in line and "Time" in lines[i + 2]:
                lines_to_remove.extend([i, i + 1])
                atoms_to_remove.append(i + 1)
            elif i + 1 < len(lines) and "Time" in line and "Time" in lines[i + 1]:
                lines_to_remove.append(i)
            elif i + 1 < len(lines) and "Time" in line and "0.0000" in lines[i + 1]:
                lines_to_remove.append(i + 1)
                atoms_to_remove.append(i + 1)

    lines_to_remove = list(set(lines_to_remove))
    atoms_to_remove = list(set(atoms_to_remove))

    if lines_to_remove:
        with open(logfile, "w", encoding="utf-8") as logf:
            for i, line in enumerate(lines):
                if i not in lines_to_remove:
                    logf.write(line)

    if atoms_to_remove:
        with ase.io.Trajectory(trajfile, "r") as traj:
            all_atoms = list(traj)
        cleaned_atoms = [
            atoms for i, atoms in enumerate(all_atoms) if i not in atoms_to_remove
        ]
        ase.io.write(trajfile, cleaned_atoms, format="traj")
