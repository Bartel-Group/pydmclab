import os
import numpy as np

from chgnet.data.dataset import StructureData, get_train_val_test_loader

from pydmclab.core.struc import StrucTools
from pydmclab.mlp.chgnet.dynamics import CHGNetRelaxer
from pydmclab.mlp.chgnet.trainer import CHGNetTrainer

DATA_DIR = os.path.join("output", "mlp-finetuning")
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

TEST_STRUC = os.path.join("cifs", "mp-18767-LiMnO2.cif")


def main():
    # First we can generate some artificial data to finetune with
    # Load a model to use to generate data labels
    model = CHGNetRelaxer(model="0.3.0", use_device="cpu").model

    # Load a parent structure
    initial_structure = StrucTools(TEST_STRUC).structure

    # Initialize lists to store artificial data
    structures, energies_per_atom, forces, stresses, magmoms = [], [], [], [], []

    # Generate data for multiple structures
    num_structures = 10
    for _ in range(num_structures):
        print(
            "Generating data for structure - ", _ + 1, " of ", num_structures, end="\r"
        )
        structure = initial_structure.copy()
        structure.apply_strain(np.random.uniform(-0.1, 0.1, size=3))
        structure.perturb(0.1)

        # Predict the structure properties
        pred = model.predict_structure(structure)

        # Add some noise to the data and store it
        structures.append(structure)
        energies_per_atom.append(pred["e"] + np.random.uniform(-0.1, 0.1, size=1))
        forces.append(pred["f"] + np.random.uniform(-0.01, 0.01, size=pred["f"].shape))
        stresses.append(
            pred["s"] * -10 + np.random.uniform(-0.05, 0.05, size=pred["s"].shape)
        )
        magmoms.append(pred["m"] + np.random.uniform(-0.03, 0.03, size=pred["m"].shape))
    print("\n")

    # Store all of the data in the StructureData object
    dataset = StructureData(
        structures=structures,
        energies=energies_per_atom,
        forces=forces,
        stresses=stresses,
        magmoms=magmoms,
    )

    # Split the data into train, validation, and test sets
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, batch_size=8, train_ratio=0.9, val_ratio=0.05
    )

    # Now, we can fine-tune a model on the generated data
    # Using the same model as before for example
    # Optionally, we can freeze most of the model layers prior to tuning
    for layer in [
        model.atom_embedding,
        model.bond_embedding,
        model.angle_embedding,
        model.bond_basis_expansion,
        model.angle_basis_expansion,
        model.atom_conv_layers[:-1],
        model.bond_conv_layers,
        model.angle_layers,
    ]:
        for param in layer.parameters():
            param.requires_grad = False

    # Initialize the trainer with the model and other hyperparameters
    trainer = CHGNetTrainer(
        model=model,
        targets="efsm",
        optimizer="Adam",
        scheduler="CosineAnnealingLR",
        criterion="MSE",
        epochs=3,
        learning_rate=1e-2,
        use_device="cpu",
        print_freq=6,
        optimizer_kwargs={},  # No need to pass any optimizer kwargs for now, but you could
        scheduler_kwargs={},  # No need to pass any scheduler kwargs for now, but you could
    )

    # Run the trainer
    trainer.train(train_loader, val_loader, test_loader, save_dir=DATA_DIR)

    # Store the best model according to validation loss for later use
    best_model = trainer.best_model

    return None


if __name__ == "__main__":
    main()
