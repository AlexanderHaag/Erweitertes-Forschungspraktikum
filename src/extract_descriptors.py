"""Calculate RDKit molecular descriptors used by the solubility models."""

import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import rdMolDescriptors


def get_mol_weight(mol) -> float:
    """Return the molecular weight of a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        The molecular weight rounded to three decimal places.
    """
    mol_weight = round(Descriptors.MolWt(mol), 3)
    return mol_weight


def get_logP(mol) -> float:
    """Return the octanol-water partition coefficient (logP) for a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        The calculated logP value rounded to five decimal places.
    """
    logP = round(Descriptors.MolLogP(mol), 5)
    return logP


def get_H_donors(mol) -> int:
    """Count the number of hydrogen-bond donors in a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        The number of hydrogen-bond donor atoms.
    """
    H_donors = Lipinski.NumHDonors(mol)
    return H_donors


def get_H_acceptors(mol) -> int:
    """Count the number of hydrogen-bond acceptors in a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        The number of hydrogen-bond acceptor atoms.
    """
    H_acceptors = Lipinski.NumHAcceptors(mol)
    return H_acceptors


def get_aromatic_ratio(mol) -> float:
    """Return the fraction of aromatic atoms in the molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        The ratio of aromatic atoms to total atoms, rounded to three decimals.
    """
    atom_count = 0
    aromatic_atom_count = 0
    for atom in mol.GetAtoms():
        atom_count += 1
        if atom.GetIsAromatic() == True:
            aromatic_atom_count += 1
    return round(aromatic_atom_count / atom_count, 3)


def get_formal_charge(mol) -> int:
    """Return the formal charge of a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        The integer formal charge of the molecule.
    """
    total_charge = Chem.GetFormalCharge(mol)
    return total_charge


def get_TPSA(mol) -> float:
    """Return the topological polar surface area (TPSA) of a molecule.

    Args:
        mol: RDKit molecule object.

    Returns:
        The TPSA value rounded to two decimal places.
    """
    tpsa = round(rdMolDescriptors.CalcTPSA(mol), 2)
    return tpsa


def get_func_atom_ratio(mol) -> float:
    """Return the ratio of non-carbon or chemically substituted atoms to all atoms.

    The function counts atoms that are not carbon, or carbon atoms attached to
    any bond type other than a single bond, as functional atoms.

    Args:
        mol: RDKit molecule object.

    Returns:
        The functional atom ratio, rounded to three decimal places.
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

    return round(func_atom_count / atom_count, 3)


def get_ring_count(mol) -> int:
    """Return the number of rings in a molecule using the smallest set of smallest rings.

    Args:
        mol: RDKit molecule object.

    Returns:
        The number of rings detected in the molecular structure.
    """
    sssr = Chem.GetSymmSSSR(mol)
    ring_count = len(sssr)
    return ring_count


def extract_descriptors(smiles) -> list:
    """Compute a set of molecular descriptors for a SMILES string.

    Args:
        smiles: A valid SMILES string representing a molecule.

    Returns:
        A list containing the calculated descriptors in the following order:
        mol_weight, logP, H_donors, H_acceptors, arom_ratio, formal_charge,
        tpsa, func_atom_ratio, and ring_count.
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
    """Run a small descriptor-extraction example."""
    test_smiles = "c1ccccc1C"
    extract_descriptors(test_smiles)

if __name__ == "__main__":
    main()