import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def main():
    # create the training + test sets
    try:
        data = pd.read_csv('Data/BaseHackathon.csv')
    except IOError:
        print("io ERROR-->Could not locate file.")
    numerics_cols = data.select_dtypes(include=[np.number])
    all_cols = set(data.columns)
    quantity_cols = set(numerics_cols.columns)

    quality_cols = all_cols - quantity_cols
    print(quality_cols)
    print(len(quality_cols))
    #for quan_col in quantity_cols:
    #    current_col = data[quan_col]
    #    print(current_col[current_col < current_col.std()*3])

main()