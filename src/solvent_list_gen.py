import extract_descriptors
import csv

def main():
    solvent_dict = {}
    with open("src/BigSolDBv2.0.csv", "r") as csvfile:
        for entries in csv.reader(csvfile, quotechar='"', delimiter=',',quoting=csv.QUOTE_ALL, skipinitialspace=True):
            solvent = entries[2]
            if solvent in solvent_dict.keys():
                solvent_dict[solvent] += 1
            else:
                solvent_dict[solvent] = 1
    solvents = sorted(solvent_dict.items(), key=lambda item:item[1], reverse=True)
    solvent_count = len(solvents)
    with open("src/solvent_list.txt", "w") as solvent_list_file:
        solvent_list_file.write(f"solvent count : {solvent_count}\n")
        for solvent, amount in solvents:
            solvent_list_file.write(f"{solvent}: {amount}\n")

if __name__=="__main__":
    main()