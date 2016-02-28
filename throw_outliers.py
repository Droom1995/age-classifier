import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import re

def throw_outliers(data, perc=75):
    """ create the training + test sets"""

    numerics_cols = data.select_dtypes(include=[np.number]).columns.values
    quantity_cols_3devs = {}
    quantity_cols_95percentiles = {}
    quantity_cols_5percentiles = {}
    quantity_cols_mean = {}
    old_data = {}

    for quan_col in numerics_cols:
        current_col = data[quan_col]
        quantity_cols_3devs[quan_col] = current_col.std()*3

        quantity_cols_95percentiles[quan_col] = np.percentile(current_col, perc)
        quantity_cols_5percentiles[quan_col] = np.percentile(current_col, 100 - perc)
        quantity_cols_mean[quan_col] = current_col.mean()
        old_data[quan_col] = current_col #len(current_col[abs(current_col) > current_col.std()*3])

    for index, row in data.iterrows():
        for col in numerics_cols:
            if row[col] > quantity_cols_mean[col]+quantity_cols_3devs[col]:
                row[col] = quantity_cols_95percentiles[col]
            elif row[col] < quantity_cols_mean[col]-quantity_cols_3devs[col]:
                row[col] = quantity_cols_5percentiles[col]

    for quan_col in numerics_cols:
        current_col = data[quan_col]
        print(old_data[quan_col], ' ', len(current_col[abs(current_col) > quantity_cols_3devs[col]]))
    return data


def concat_free_money(data):
    # for col in data.columns:
    #     print(col)
    fm_col_tag = 'FREE_MONEY'
        # print(value)
    for col in data.columns:
        if col.split('_')[-1].startswith('FM'):
            if fm_col_tag in data:
                data[fm_col_tag] = data[fm_col_tag] + data[col]
                data = data.drop(col, 1)
            else:
                data[fm_col_tag] = data[col]
                data = data.drop(col, 1)
    # print('-------------------')
    # for value in data['FREE_MONEY']:
    #     print(value)
    return data

if __name__ == '__main__':
    import time
    from pandas import read_csv, DataFrame
    t = time.time()
    dataset1 = read_csv('Data/BaseHackathon.csv')
    dataset1 = dataset1.head(1000)
    dataset1 = dataset1.select_dtypes(include=[np.number])
    use_field = list(dataset1.columns.values)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(dataset1)
    dataset1 = imp.transform(dataset1)
    dataset1 = DataFrame(dataset1, columns=use_field)
    concat_free_money(dataset1)
    print(float(time.time() - t)*100/60)