import extract_descriptors
import csv

def main():
    """This function generates a csv file from parameters calculated by "extract.parameters" and directly from "BigSolDBv2.0.csv". 
    These parameters are then saved in a new csv file.
    """
    solvent_list = ["ethanol",
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
    
    counter = 0

    with open("src/BigSolDBv2.0.csv", "r") as csvfile:
        for entries in csv.reader(csvfile, quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True):
            smiles = entries[0]
            temp = entries[1]
            solvent = entries[2]
            # solubility = entries[5]
            logS = entries[6]
            if logS == "":
                continue
            if solvent not in solvent_list: #total entries left: 64777
                continue
            #print(f"{smiles}, {temp}, {solvent}, {logS}")
            counter +=1
            params = extract_descriptors.extract_descriptors(smiles)
        print(counter)

if __name__=="__main__":
    main()
#extract_parameters()