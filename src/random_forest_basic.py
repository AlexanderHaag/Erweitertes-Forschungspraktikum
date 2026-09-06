import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def main():

    TEST_SIZE = 0.1
    RANDOM_STATE = 42
    N_ESTIMATORS = 100
    read_file = "src/data_basic.csv"

    df = pd.read_csv(read_file)
    print(df.shape)
    print(df.head())

    parameters = ["mol_weight", 
                  "logP", 
                  "H_donors", 
                  "H_acceptors", 
                  "arom_ratio", 
                  "formal_charge", 
                  "tpsa", 
                  "func_atom_ratio", 
                  "ring_count", 
                  "temp", 
                  "solvent"]
    
    target = "logS"

    X = df[parameters]
    Y = df[target]

    numeric_params = ["mol_weight", 
                  "logP", 
                  "H_donors", 
                  "H_acceptors", 
                  "arom_ratio", 
                  "formal_charge", 
                  "tpsa", 
                  "func_atom_ratio", 
                  "ring_count", 
                  "temp"]

    categorical_params = ["solvent"]

    preprocessor = ColumnTransformer([
    ("numeric", "passthrough", numeric_params),
    ("solvent", OneHotEncoder(handle_unknown="ignore"), categorical_params)
    ])
    model = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
        )

    pipeline.fit(X_train, Y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(Y_test, predictions)
    rmse = root_mean_squared_error(Y_test, predictions)
    r2 = r2_score(Y_test, predictions)

    with open("src/results.txt", "a") as resultfile:
        resultfile.write(f"BASIC MODEL: Test size: {TEST_SIZE}; N_ESTIMATORS: {N_ESTIMATORS}\n")
        resultfile.write(f"\tMAE: {mae}\n")
        resultfile.write(f"\tRMSE: {rmse}\n")
        resultfile.write(f"\tr2: {r2}\n\n")
    return  0

if __name__=="__main__":
    main()
