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

dataset1 = read_csv('Data/BaseHackathon.csv')
dataset2 = read_csv('Data/y.csv')
# dataset2 = read_csv('Data/BaseHackathon.csv')

# print(dataset2)

# merge = dataset2.join(dataset1, on='SUBS_ID')

# m = pd.merge(dataset1, dataset2, on='SUBS_ID', how='left')
# m.to_csv(open('Data/X.csv', 'w'))
# dataset = dataset1
import traceback

imp = Imputer(missing_values='NA', strategy='mean', axis=0)
imp.fit(dataset1)
dataset1 = dataset1.select_dtypes(include=[np.number])
train = dataset1[:int(len(dataset1)/3*2)]
test = dataset1[int(len(dataset1)/3*2):]
target_1 = dataset2.AGE_GROUP1[:int(len(dataset2)/3*2)]
target_test_1 = dataset2.AGE_GROUP1[int(len(dataset2)/3*2):]

model = ExtraTreesClassifier()
model.fit(train, target_1)
# display the relative importance of each attribute
print(model.feature_importances_)
#
#
#
#
# #
# # X = [
# #      [x if not np.isnan(x) else -1 for x in train.TECH_PRICE_PLAN_BOP]
# # ]
# # X_t = [
# #        [x if not np.isnan(x) else -1 for x in test.TECH_PRICE_PLAN_BOP],
# # ]
# # X_full = [
# #        [x if not np.isnan(x) else -1 for x in dataset.TECH_PRICE_PLAN_BOP],
# # ]
# # # test_predict = [
# # #     [x if not np.isnan(x) else -1 for x in test_dataset.Pclass]
# # # ]
# # X = DataFrame(np.array(X).T)
# # X_t = DataFrame(np.array(X_t).T)
# # X_full = DataFrame(np.array(X_full).T)
# # X_f_t = []
# # # X_f_t = DataFrame(np.array(test_predict).T)
