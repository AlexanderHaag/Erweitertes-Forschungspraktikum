import extractparams
import csv

with open("BigSolDBv2.0.csv", "r") as csvfile:
    for entries in csv.reader(csvfile, quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True):
        smiles = entries[0]
        temp = entries[1]
        solvent = entries[2]
        # solubility = entries[5]
        logS = entries[6]
        if logS == "":
            continue
        print(f"{smiles}, {temp}, {solvent}, {logS}")
        params = extractparams.extractparams(smiles)

#extract_parameters()