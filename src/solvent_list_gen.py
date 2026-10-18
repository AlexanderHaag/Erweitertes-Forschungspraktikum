import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "BigSolDBv2.0.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "solvent_list.txt"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():
    solvent_dict = {}
    with RAW_DATA_PATH.open("r", newline="") as csvfile:
        for entries in csv.reader(csvfile, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True):
            solvent = entries[2]
            if solvent in solvent_dict.keys():
                solvent_dict[solvent] += 1
            else:
                solvent_dict[solvent] = 1
    solvents = sorted(solvent_dict.items(), key=lambda item: item[1], reverse=True)
    solvent_count = len(solvents)
    with OUTPUT_PATH.open("w", encoding="utf-8") as solvent_list_file:
        solvent_list_file.write(f"solvent count : {solvent_count}\n")
        for solvent, amount in solvents:
            solvent_list_file.write(f"{solvent}: {amount}\n")


if __name__ == "__main__":
    main()