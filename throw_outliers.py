import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt

def throw_outliers(data):
    # create the training + test sets
    try:
        data = pd.read_csv('Data/BaseHackathon.csv')
    except IOError:
        print("io ERROR-->Could not locate file.")

    dataset1 = data.select_dtypes(include=[np.number])
    use_field = list(dataset1.columns.values)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(dataset1)
    data = imp.transform(dataset1)
    data = pd.DataFrame(data)
    # data = data[:10000] # for test
    numerics_cols = data.select_dtypes(include=[np.number])
    all_cols = set(data.columns)
    quantity_cols = set(numerics_cols.columns)

    quality_cols = all_cols - quantity_cols

    #quantity_cols = list(quantity_cols)[:] # for test

    quantity_cols_3devs = {}
    quantity_cols_95percentiles = {}
    quantity_cols_5percentiles = {}
    old_data = {}
    for quan_col in quantity_cols:
        current_col = data[quan_col]

        quantity_cols_3devs[quan_col] = current_col.std()*3
        quantity_cols_95percentiles[quan_col] = np.percentile(current_col,75)
        quantity_cols_5percentiles[quan_col] = np.percentile(current_col,25)
        old_data[quan_col] = current_col #len(current_col[abs(current_col) > current_col.std()*3])


    for index, row in data.iterrows():
        for col in quantity_cols:
            if row[col] > quantity_cols_3devs[col]:
                row[col] = quantity_cols_95percentiles[col]
            elif row[col] < quantity_cols_3devs[col]:
                row[col] = quantity_cols_5percentiles[col]

    for quan_col in quantity_cols:
        current_col = data[quan_col]
        print(old_data[quan_col], ' ', len(current_col[abs(current_col) > quantity_cols_3devs[col]]))
    return data
