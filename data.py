from pandas import read_csv, DataFrame
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import itertools as it
dataset1 = read_csv('Data/Target_AgeGroup.csv')
dataset2 = read_csv('Data/BaseHackathon.csv')

print(dataset2)

# merge = dataset2.join(dataset1, on='SUBS_ID')

m = pd.merge(dataset1, dataset2, on='SUBS_ID', how='left')
m.to_csv(open('Data/X.csv', 'w'))
# dataset = dataset1


#
# train = dataset[:int(len(dataset)/3*2)]
# test = dataset[int(len(dataset)/3*2):]
# target = dataset.Survived[:int(len(dataset)/3*2)]
# target_test = dataset.Survived[int(len(dataset)/3*2):]
#
# X = [
#      [x if not np.isnan(x) else -1 for x in train.TECH_PRICE_PLAN_BOP]
# ]
# X_t = [
#        [x if not np.isnan(x) else -1 for x in test.TECH_PRICE_PLAN_BOP],
# ]
# X_full = [
#        [x if not np.isnan(x) else -1 for x in dataset.TECH_PRICE_PLAN_BOP],
# ]
# # test_predict = [
# #     [x if not np.isnan(x) else -1 for x in test_dataset.Pclass]
# # ]
# X = DataFrame(np.array(X).T)
# X_t = DataFrame(np.array(X_t).T)
# X_full = DataFrame(np.array(X_full).T)
# X_f_t = []
# # X_f_t = DataFrame(np.array(test_predict).T)