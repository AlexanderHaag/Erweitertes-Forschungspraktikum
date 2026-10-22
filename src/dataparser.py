"""Build processed solubility datasets with canonical solute and solvent IDs."""

import csv
from pathlib import Path

import extract_descriptors
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def parse_data():
    """Prepare cleaned solubility datasets from the raw database.

    This script reads the source file ``data/raw/BigSolDBv2.0.csv``, filters out
    incomplete entries and unsupported solvents, computes molecular descriptors
    for each molecule via the RDKit utilities in ``extract_descriptors``, and
    writes the processed data to CSV files for model training.

    The function creates three output datasets:

    1. ``data/processed/data_basic.csv``
         Contains molecule descriptors, ``temp``, ``solvent``, ``logS``, and a
         canonical ``molecule_id`` and ``solvent_id`` used to group measurements
         by molecule-solvent pair when splitting model data.

    2. ``data/processed/data_sol_desc.csv``
    Contains the same molecule and solvent identities and descriptors as well as
    descriptor values for the solvent itself, allowing solvent-aware learning.

    3. ``data/processed/data_all_entries.csv``
    Contains every row with a valid ``logS`` value, without filtering by solvent,
    using the same solute and solvent descriptor columns as ``data_sol_desc.csv``.

    Returns:
        None: All three datasets are written directly to disk during execution.
    """
    solvent_whitelist = ["ethanol",
                        "methanol",
                        "isopropanol",
                        "water",
                        "ethyl acetate",
                        "acetone",
                        "acetonitrile",
                        "DMF",
                        "toluene",
                        "1,4-dioxane",
                        "THF",
                        "n-hexane",
                        "cyclohexane",
                        "DMSO",
                        "chloroform",
                        "dichloromethane"]

    source_csv = RAW_DATA_DIR / "BigSolDBv2.0.csv"
    basic_output = PROCESSED_DATA_DIR / "data_basic.csv"
    all_entries_output = PROCESSED_DATA_DIR / "data_all_entries.csv"

    counter = 0

    with source_csv.open("r", newline="") as csvfile, basic_output.open("w", newline="") as datafile:
        datafile_header = "mol_weight,logP,H_donors,H_acceptors,arom_ratio,formal_charge,tpsa,func_atom_ratio,ring_count,temp,solvent,logS,solvent_id,molecule_id\n"
        datafile.write(datafile_header)
        reader = csv.reader(
            csvfile,
            quotechar='"',
            delimiter=',',
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
        )
        next(reader, None)
        for entries in reader:
            smiles = entries[0]
            temp = entries[1]
            solvent = entries[2]
            logS = entries[6]
            if logS == "":
                continue
            if solvent not in solvent_whitelist:  # total entries left: 64777
                continue
            if solvent == "1,4-dioxane":
                solvent = "dioxane"
            counter += 1
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid molecule SMILES: {smiles!r}")
            solvent_molecule = Chem.MolFromSmiles(entries[3])
            if solvent_molecule is None:
                raise ValueError(f"Invalid solvent SMILES: {entries[3]!r}")
            molecule_id = Chem.MolToSmiles(molecule, canonical=True)
            solvent_id = Chem.MolToSmiles(solvent_molecule, canonical=True)
            params = extract_descriptors.extract_descriptors(smiles)
            # Keep solvent identity separate from its human-readable name.
            param_str = (
                ",".join(str(param) for param in params)
                + f",{temp},{solvent},{logS},{solvent_id},{molecule_id}\n"
            )
            datafile.write(param_str)
            print(counter)

    solvent_output = PROCESSED_DATA_DIR / "data_sol_desc.csv"
    counter = 0

    with source_csv.open("r", newline="") as csvfile, solvent_output.open("w", newline="") as datafile2:
        datafile2_header = "mol_weight,logP,H_donors,H_acceptors,arom_ratio,formal_charge,tpsa,func_atom_ratio,ring_count,temp,logS,sol_mol_weight,sol_logP,sol_H_donors,sol_H_acceptors,sol_arom_ratio,sol_formal_charge,sol_tpsa,sol_func_atom_ratio,sol_ring_count,solvent_id,molecule_id\n"
        datafile2.write(datafile2_header)
        reader = csv.reader(
            csvfile,
            quotechar='"',
            delimiter=',',
            quoting=csv.QUOTE_ALL,
            skipinitialspace=True,
        )
        next(reader, None)
        for entries in reader:
            smiles = entries[0]
            temp = entries[1]
            solvent = entries[2]
            solvent_smiles = entries[3]
            logS = entries[6]
            if logS == "":
                continue
            if solvent not in solvent_whitelist:  # total entries left: 64777
                continue
            if solvent == "1,4-dioxane":
                solvent = "dioxane"
            counter += 1
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid molecule SMILES: {smiles!r}")
            molecule_id = Chem.MolToSmiles(molecule, canonical=True)
            solvent_molecule = Chem.MolFromSmiles(solvent_smiles)
            if solvent_molecule is None:
                raise ValueError(f"Invalid solvent SMILES: {solvent_smiles!r}")
            solvent_id = Chem.MolToSmiles(solvent_molecule, canonical=True)
            params = extract_descriptors.extract_descriptors(smiles)
            sol_params = extract_descriptors.extract_descriptors(solvent_smiles)
            # Descriptors are model inputs; the canonical solvent ID is split metadata.
            param_str = (
                ",".join(str(param) for param in params)
                + f",{temp},{logS},"
                + ",".join(str(param) for param in sol_params)
                + f",{solvent_id},{molecule_id}\n"
            )
            datafile2.write(param_str)
            print(counter)

    all_entries_header = [
        "mol_weight",
        "logP",
        "H_donors",
        "H_acceptors",
        "arom_ratio",
        "formal_charge",
        "tpsa",
        "func_atom_ratio",
        "ring_count",
        "temp",
        "logS",
        "sol_mol_weight",
        "sol_logP",
        "sol_H_donors",
        "sol_H_acceptors",
        "sol_arom_ratio",
        "sol_formal_charge",
        "sol_tpsa",
        "sol_func_atom_ratio",
        "sol_ring_count",
        "solvent_id",
        "molecule_id",
    ]

    counter = 0

    with source_csv.open("r", newline="", encoding="utf-8") as csvfile:
        with all_entries_output.open("w", newline="", encoding="utf-8") as datafile3:
            reader = csv.DictReader(csvfile)
            writer = csv.writer(datafile3)
            writer.writerow(all_entries_header)

            for entries in reader:
                logS = entries["LogS(mol/L)"]
                if not logS or not logS.strip():
                    continue

                smiles = entries["SMILES_Solute"]
                temp = entries["Temperature_K"]
                solvent = entries["Solvent"]
                solvent_smiles = entries["SMILES_Solvent"]

                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise ValueError(f"Invalid solute SMILES: {smiles!r}")
                solvent_molecule = Chem.MolFromSmiles(solvent_smiles)
                if solvent_molecule is None:
                    raise ValueError(f"Invalid solvent SMILES for {solvent!r}: {solvent_smiles!r}")

                molecule_id = Chem.MolToSmiles(molecule, canonical=True)
                solvent_id = Chem.MolToSmiles(solvent_molecule, canonical=True)
                params = extract_descriptors.extract_descriptors(smiles)
                sol_params = extract_descriptors.extract_descriptors(solvent_smiles)
                # Preserve exact solvent identity for pair-level validation.
                writer.writerow(
                    [*params, temp, logS, *sol_params, solvent_id, molecule_id]
                )
                counter += 1

    print(f"All-entry rows written: {counter}")


parse_data()
