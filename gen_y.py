from pandas import read_csv, DataFrame
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

# dataset1 = read_csv('Data/BaseHackathon.csv')
dataset1 = read_csv('Data/Target_AgeGroup.csv')
# dataset2 = read_csv('Data/y.csv')
dataset2 = read_csv('Data/BaseHackathon.csv')

# print(dataset2)

m = pd.merge(dataset1, dataset2, on='SUBS_ID', how='left')
m.to_csv(open('Data/X.csv', 'w'))
import csv
r = csv.DictReader(open('Data/X.csv', 'r'))
w = csv.DictWriter(open('Data/y.csv', 'w'), fieldnames=['AGE_GROUP1', 'AGE_GROUP2'])
w.writeheader()
for x in r:
    w.writerow({'AGE_GROUP1': x['AGE_GROUP1'], 'AGE_GROUP2': x['AGE_GROUP2']})
