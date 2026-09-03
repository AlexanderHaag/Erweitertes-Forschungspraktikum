import extract_descriptors
import csv

def main():
    """This function generates a csv file from parameters calculated by "extract.parameters" and directly from "BigSolDBv2.0.csv". 
    These parameters are then saved in "data.csv".
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
    
    counter = 0

    with open("src/BigSolDBv2.0.csv", "r") as csvfile, open("src/data.csv", "w") as datafile:
        datafile_header = "mol_weight,logP,H_donors,H_acceptors,arom_ratio,formal_charge,tpsa,func_atom_ratio,ring_count,temp,solvent,logS\n"
        datafile.write(datafile_header)
        for entries in csv.reader(csvfile, quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True):
            smiles = entries[0]
            temp = entries[1]
            solvent = entries[2]
            logS = entries[6]
            if logS == "":
                continue
            if solvent not in solvent_whitelist: #total entries left: 64777
                continue
            if solvent == "1,4-dioxane":
                solvent = "dioxane"
            counter += 1
            params = extract_descriptors.extract_descriptors(smiles)
            param_str = ",".join(str(param) for param in params) + f",{temp},{solvent},{logS}\n"
            datafile.write(param_str)
            print(counter)

if __name__=="__main__":
    main()
