import csv
from pathlib import Path

import extract_descriptors

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

    The function creates two output datasets:

    1. ``data/processed/data_basic.csv``
       Contains molecule descriptors plus ``temp``, ``solvent``, and ``logS``.

    2. ``data/processed/data_sol_desc.csv``
       Contains the same molecule descriptors as well as descriptor values for
       the solvent itself, allowing solvent-aware learning.

    Returns:
        None: Both datasets are written directly to disk during execution.
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

    counter = 0

    with source_csv.open("r", newline="") as csvfile, basic_output.open("w", newline="") as datafile:
        datafile_header = "mol_weight,logP,H_donors,H_acceptors,arom_ratio,formal_charge,tpsa,func_atom_ratio,ring_count,temp,solvent,logS\n"
        datafile.write(datafile_header)
        for entries in csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
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
            params = extract_descriptors.extract_descriptors(smiles)
            param_str = ",".join(str(param) for param in params) + f",{temp},{solvent},{logS}\n"
            datafile.write(param_str)
            print(counter)

    solvent_output = PROCESSED_DATA_DIR / "data_sol_desc.csv"
    counter = 0

    with source_csv.open("r", newline="") as csvfile, solvent_output.open("w", newline="") as datafile2:
        datafile2_header = "mol_weight,logP,H_donors,H_acceptors,arom_ratio,formal_charge,tpsa,func_atom_ratio,ring_count,temp,logS,sol_mol_weight,sol_logP,sol_H_donors,sol_H_acceptors,sol_arom_ratio,sol_formal_charge,sol_tpsa,sol_func_atom_ratio,sol_ring_count\n"
        datafile2.write(datafile2_header)
        for entries in csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
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
            params = extract_descriptors.extract_descriptors(smiles)
            sol_params = extract_descriptors.extract_descriptors(solvent_smiles)
            param_str = ",".join(str(param) for param in params) + f",{temp},{logS}," + ",".join(str(param) for param in sol_params) + "\n"
            datafile2.write(param_str)
            print(counter)


parse_data()
