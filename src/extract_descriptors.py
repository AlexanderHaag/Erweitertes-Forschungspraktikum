import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors


def get_mol_weight(mol) -> float:
    """Calculates molecular weight using rdkit.Chem.Descriptors module

    Args:
        mol (mol)): Input molecule

    Returns:
        float: Returns molecular weight of input molecule
    """
    mol_weight = round(Descriptors.MolWt(mol), 3)
    return mol_weight

def get_logP(mol) -> float:
    """Calculates logP of input molecule using the rdkit.Chem.Descriptors module

    Args:
        mol (mol): Input molecule

    Returns:
        float: Calculated logP of input molecule
    """
    logP = round(Descriptors.MolLogP(mol), 5)
    return logP

def get_H_donors(mol) -> int:
    """Counts H donors in input molecule using rdkit.Chem.lipinski module

    Args:
        mol (mol): Input molecule

    Returns:
        int: Number of H donors present in input  molecule
    """
    H_donors = Lipinski.NumHDonors(mol)
    return H_donors

def get_H_acceptors(mol) -> int:
    """Counts H acceptors in input molecule using rdkit.Chem.lipinski module

    Args:
        mol (mol): Input molecule

    Returns:
        int: Number of H acceptors present in input  molecule
    """
    H_acceptors = Lipinski.NumHAcceptors(mol)
    return H_acceptors

def get_aromatic_ratio(mol) -> float:
    """Calculates the ratio of aromatic heavy atoms to total atoms

    Args:
        mol (mol): Input molecule

    Returns:
        float: Aromatic atom ratio (0 <= x <= 1)
    """
    atom_count = 0
    aromatic_atom_count = 0
    for atom in mol.GetAtoms():
        atom_count += 1
        if atom.GetIsAromatic() == True:
            aromatic_atom_count += 1
    return round(aromatic_atom_count/atom_count, 3)

def get_formal_charge(mol) -> int:
    """Calculates formal charge of input molecule using the rdkit.Chem module

    Args:
        mol (mol): Input molecule

    Returns:
        int: Formal charge of input molecule
    """
    total_charge = Chem.GetFormalCharge(mol)
    return total_charge

def get_TPSA(mol) -> float:
    """Calculates TPSA of input molecule using the rdkit.Chem.rdMolDescriptors module

    Args:
        mol (mol): Input molecule

    Returns:
        float: TPSA (topological polar surface area) of input molecule
    """
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
    return tpsa

def get_func_atom_ratio(mol) -> float:
    """Calculates the ratio of atom in functional groups to the total amount of atoms in the input molecule

    Args:
        mol (mol): Input molecule

    Returns:
        float: Functional atom ratio (0 <= x <= 1)
    """
    atom_count = 0
    func_atom_count = 0
    for atom in mol.GetAtoms():
        atom_count += 1
        if atom.GetSymbol() != "C":
            func_atom_count += 1
            continue
        elif any(bond.GetBondType() != Chem.BondType.SINGLE for bond in atom.GetBonds()):
                func_atom_count += 1

    return round(func_atom_count/atom_count, 3)

def get_ring_count(mol) -> int:
    """Calculates smallest set of smallest rings

    Args:
        mol (mol): Input molecule

    Returns:
        int: SSSR
    """
    sssr = Chem.GetSymmSSSR(mol)
    ring_count = len(sssr)
    return ring_count

def extract_descriptors(smiles) -> list:
    """Calculates parameters of input Smiles

    Args:
        smiles (str): Smiles code of input molecule

    Returns:
        list: List of the calculated parameters
    """
    mol = Chem.MolFromSmiles(smiles)
    mol_weight = get_mol_weight(mol)
    logP = get_logP(mol)
    H_donors = get_H_donors(mol)
    H_acceptors = get_H_acceptors(mol)
    arom_ratio = get_aromatic_ratio(mol)
    formal_charge = get_formal_charge(mol)
    tpsa = get_TPSA(mol)
    func_atom_ratio = get_func_atom_ratio(mol)
    ring_count = get_ring_count(mol)

    descriptors = [mol_weight, logP, H_donors, H_acceptors, arom_ratio, formal_charge, tpsa, func_atom_ratio, ring_count]
    # print(f"mol_weight: {mol_weight}\nlogP: {logP}\nH_donors: {H_donors}\nH_acceptors: {H_acceptors}\narom_ratio: {arom_ratio}\nformal_charge: {formal_charge}\ntpsa: {tpsa}\nfunc_atom_ratio: {func_atom_ratio}\nring_count: {ring_count}")
    return descriptors



def main():
    test_smiles = "c1ccccc1C"
    extract_descriptors(test_smiles)

if __name__ == "__main__":
    main()