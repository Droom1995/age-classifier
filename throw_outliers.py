import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
from sklearn import preprocessing
import re

def throw_outliers(data, perc=99):
    """ create the training + test sets"""

    numerics_cols = data.select_dtypes(include=[np.number]).columns.values
    quantity_cols_3devs = {}
    quantity_cols_95percentiles = {}
    quantity_cols_5percentiles = {}
    quantity_cols_mean = {}
    old_data = {}
#     cols_leave = ['ON_NET_INCOMING_CALLS_COUNT',
#     'TOTAL_INCOMING_CALLS_COUNT',
#     'ON_NET_INCOMING_CALLS_MIN',
#     'TOTAL_INCOMING_CALLS_MIN',
#     'TOTAL_OUTGOING_CALLS_COUNT',
#     'ON_NET_OUTGOING_CALLS_COUNT_BD',
#     'ON_NET_OUTGOING_CALLS_COUNT',
#     'ON_NET_OUTGOING_CALLS_COUNT_WE',
#     'ON_NET_OUTGOING_CALLS_MIN_WE',
#     'ON_NET_OUTGOING_CALLS_MIN_BD',
#     'TOTAL_OUTGOING_CALLS_MIN',
#     'ON_NET_OUTGOING_CALLS_MIN',
#     'PERIODIC_FEES_CORE',
#     'OFF_NET_INCOMING_CALLS_COUNT',
#     'ON_NET_OUTGOING_REVENUE_CORE',
#     'OFF_NET_INCOMING_CALLS_MIN',
#     'TOTAL_SMS_REVENUE',
#     'FREE_MONEY',
#     'ON_NET_OUT_REVENUE_CORE_BD',
# 'OFF_NET_SMS_REVENUE',
# 'TOTAL_OUTGOING_REVENUE_CORE',
# 'CHARGE_WO_TAX_CORE',
# 'TOTAL_CHARGE_CORE',
# 'ON_NET_OUT_REVENUE_CORE_WE',
# 'OFF_NET_ITC_REVENUE',
# 'REFILLED_AMOUNT',
# 'OFF_NET_OUTGOING_CALLS_COUNT',
# 'OFF_NET_OUTGOING_REVENUE_CORE',
# 'OFF_NET_OUT_REVENUE_CORE_BD',
# 'OFF_NET_OUT_CALLS_COUNT_BD',
# 'OFF_NET_OUTGOING_CALLS_MIN_BD',
# 'OFF_NET_OUTGOING_CALLS_MIN',
# 'OFF_NET_ITC_COST',
# 'OTHER_OUTGOING_CALLS_MIN',
# 'OTHER_VAS_REVENUE_CORE',
# 'MOBILE_DATA_REVENUE_CORE',
# 'OFF_NET_OUTGOING_CALLS_MIN_WE',
# 'INTERNATIONAL_SMS_REVENUE',
# 'MOBILE_DATA_MB',
# 'OFF_NET_OUT_REVENUE_CORE_WE',
# 'OFF_NET_SMS_COST',
# 'OUTGOING_SMS_COUNT_BD',
# 'SMS_REVENUE_ON_NET_CORE',
# 'INTERNET_IND',
# 'TOTAL_OUTGOING_REVENUE_FM3',
# 'INTERNATIONAL_ITC_REVENUE',
# 'SMS_IND',
# 'REFILLED_SUBS_IND',
# 'SMS_REVENUE_OFF_NET_CORE',
# 'ON_NET_OUTGOING_REVENUE_FM3',
# 'VOICE_OFF_NET_IND',
# 'REFILLED_AMOUNT_FM3',
# 'TOTAL_OUTGOING_REVENUE_FM2',
# 'OTHER_VAS_REVENUE_FM3',
# 'OFF_NET_OUTGOING_REVENUE_FM',
# 'VIMPELCOM_ITC_CALLS_COUNT_REV',
# 'ON_NET_OUTGOING_REVENUE_FM4',
# 'INT_OUTGOING_CALLS_COUNT',
# 'OUTGOING_SMS_INT_COUNT'
#     ]
#     cols_leave = set(cols_leave)
    # for col in data.columns:
        # if not(col in cols_leave):
            # data = data.drop(col, 1)
    for quan_col in numerics_cols:
        current_col = data[quan_col]
        quantity_cols_3devs[quan_col] = current_col.std()*3

        quantity_cols_95percentiles[quan_col] = np.percentile(current_col, perc)
        quantity_cols_5percentiles[quan_col] = np.percentile(current_col, 100 - perc)
        quantity_cols_mean[quan_col] = current_col.mean()
        old_data[quan_col] = current_col #len(current_col[abs(current_col) > current_col.std()*3])
        # plt.hist(current_col)
        # plt.title(quan_col)
        # plt.show()
        #data=data.drop(quan_col, 1)
        if quantity_cols_5percentiles[quan_col]>0.9 or quantity_cols_95percentiles[quan_col]<0.001 :
            print(quan_col,quantity_cols_5percentiles[quan_col],quantity_cols_95percentiles[quan_col])
            # plt.hist(current_col)
            # plt.title(quan_col)
            # plt.show()
            data=data.drop(quan_col, 1)


    # for index, row in data.iterrows():
    #     for col in numerics_cols:
    #         if row[col] > quantity_cols_mean[col]+quantity_cols_3devs[col]:
    #             row[col] = quantity_cols_95percentiles[col]
    #         elif row[col] < quantity_cols_mean[col]-quantity_cols_3devs[col]:
    #             row[col] = quantity_cols_5percentiles[col]

    # for quan_col in numerics_cols:
    #     current_col = data[quan_col]
    #     print(old_data[quan_col], ' ', len(current_col[abs(current_col) > quantity_cols_3devs[col]]))
    return data


def concat_free_money(data):
    # for col in data.columns:
    #     print(col)
    #data = throw_outliers(data,95)
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
    x = data[fm_col_tag].values.astype(float)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    data[fm_col_tag] = x_scaled
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
    #dataset1 = concat_free_money(dataset1)
    dataset1 = dataset1.drop('MonthAgo',1)
    dataset1 = throw_outliers(dataset1)
    print(float(time.time() - t)*100/60)