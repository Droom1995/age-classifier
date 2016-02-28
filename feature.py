from pandas import read_csv, DataFrame
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import itertools as it
from sklearn import metrics
import seaborn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer

dataset1 = read_csv('Data/BaseHackathon.csv')
dataset2 = read_csv('Data/y.csv')

imp = Imputer(missing_values='NA', strategy='mean', axis=0)
dataset1 = dataset1.select_dtypes(include=[object])
dataset1.drop('ACTIVITY_SMS_DAYS_IND', axis=1)
main_map = {}
# print(len(dataset1))
# dataset1 = dataset1.head(5000)
import time
t =time.time()
X = {}
ids = {}

for index, row in dataset1.iterrows():
    ids[row.SUBS_ID] = {x: [] for x in dataset1.columns.values if x != 'SUBS_ID'}

for index, row in dataset1.iterrows():
    for k, v in row.items():
        if k != 'SUBS_ID':
            ids[row.SUBS_ID][k].append(v)

    # temp = {row.SUBS_ID : index}
    # ids.update(temp)

for column in dataset1:
    map = dict(Counter(dataset1[column]))
    if map.get(np.nan) is not None:
        del(map[np.nan])
    main_map[column] = map

classes = {k: {x: i for i, x in enumerate(v)} for k, v in main_map.items()}
main_map = {k: max(v, key=v.get) for k, v in main_map.items()}

# print(classes)
# print('--------------')
d_set = set(dataset1['SUBS_ID'])
# count = len(ids)
for j, id in enumerate(d_set):
    print(j)
    rows = ids[id]
    X_t = {}

    for i, column in enumerate(rows):
        print(i)
        map = dict(Counter(rows[column]))
        t_d = {x: y for x, y in map.items() if str(x) != 'nan'}
        if not t_d:
            max_key = main_map[column]
        else:
            max_key = max(t_d, key=map.get)
        # print(max_key)
        X_t[column] = classes[column].get(max_key)
    # print(temp)
        # print(map, max(map.values()))

        # classes = []
        # mean = sum(map.values())/len(map)
        # print(map, mean)
    X.update({id: X_t})


import csv
w = csv.DictWriter(open('Data/X2.csv', 'w'), fieldnames=dataset1.columns.values)
w.writeheader()
for k, v in X.items():
    d = {x: y for x, y in v.items()}
    d['SUBS_ID'] = k
    w.writerow(d)

print((time.time() -t)/60*100)