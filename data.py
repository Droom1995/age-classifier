from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import itertools as it
from sklearn import metrics, preprocessing
import seaborn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Imputer
from throw_outliers import concat_free_money
dataset = read_csv('Data/BaseHackathon.csv')
# dataset2 = read_csv('Data/y.csv')

dataset2 = read_csv('Data/Target_AgeGroup.csv')
del dataset['MonthAgo']
# print(dataset.columns.values)

# print(dataset2)

# merge = dataset2.join(dataset1, on='SUBS_ID')


# m.to_csv(open('Data/X.csv', 'w'))
# dataset = dataset1
import traceback




dataset1 = dataset.select_dtypes(include=[np.number])
use_field = list(dataset1.columns.values)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(dataset1)
dataset1 = imp.transform(dataset1)
dataset1 = DataFrame(dataset1, columns=use_field)
dataset1 = concat_free_money(dataset1)
dataset1 = dataset1.assign(SUBS_ID=dataset.SUBS_ID)
dataset1 = pd.merge(dataset1, dataset2, on='SUBS_ID', how='left')
dataset1 = pd.merge(dataset1, read_csv('Data/X2.csv'), on='SUBS_ID', how='left')
# print(use_field)
gr = dataset1.groupby('SUBS_ID')
dataset = gr.mean()
dataset1 = dataset.copy()
dataset1 = dataset1.drop(['AGE_GROUP1', 'AGE_GROUP2'], axis=1)
print(len(dataset1.columns.values))
# dataset1 = dataset1['SUBS_ID']
      # (dataset.columns.values[0]))
# dataset1 = preprocessing.scale(dataset1)
#
train = dataset1[:int(len(dataset1)/3*2)]
test = dataset1[int(len(dataset1)/3*2):]
target_1 = dataset.AGE_GROUP2[:int(len(dataset2)/3*2)]
target_test_1 = dataset.AGE_GROUP2[int(len(dataset2)/3*2):]

model = RandomForestClassifier(n_estimators=300, n_jobs=5)
model.fit(train, target_1)
print(metrics.classification_report(target_test_1, model.predict(test)))
# display the relative importance of each attribute
# # width = 0.35
feature = model.feature_importances_
#
# # _f = [[x, y] for x, y in zip(use_field, feature)]
_f = {x: y for x, y in zip(use_field, feature)}
# # _f.sort(key = lambda x: x[1], reverse=True)
import openpyxl
wb = openpyxl.load_workbook('Data/columns_description.xlsx')
w = wb.active
for x in range(2, 178):
    w['D%s' % (x + 2)] = _f.get(w['B%s' % (x + 2)].value, -1)*100
wb.save('Data/columns_description.xlsx')
# #
# # feature = [x[1] for x in _f[:15]]
# # use_field = [x[0] for x in _f[:15]]
# # plt.bar(np.arange(len(feature)), feature, width=0.35)
# # plt.xticks(np.arange(len(feature)) + width/2., use_field)
# # plt.show()
# #
# #
# # #
# # #
# # #
# # #
# # #
# # # X = [
# # #      [x if not np.isnan(x) else -1 for x in train.TECH_PRICE_PLAN_BOP]
# # # ]
# # # X_t = [
# # #        [x if not np.isnan(x) else -1 for x in test.TECH_PRICE_PLAN_BOP],
# # # ]
# # # X_full = [
# # #        [x if not np.isnan(x) else -1 for x in dataset.TECH_PRICE_PLAN_BOP],
# # # ]
# # # # test_predict = [
# # # #     [x if not np.isnan(x) else -1 for x in test_dataset.Pclass]
# # # # ]
# # # X = DataFrame(np.array(X).T)
# # # X_t = DataFrame(np.array(X_t).T)
# # # X_full = DataFrame(np.array(X_full).T)
# # # X_f_t = []
# # # # X_f_t = DataFrame(np.array(test_predict).T)
