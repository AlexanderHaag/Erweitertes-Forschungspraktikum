import rdkit.Chem 

def extractparams(smiles):
    
    mol = rdkit.Chem.MolFromSmiles(smiles)

    return 0