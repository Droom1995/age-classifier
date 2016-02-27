import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

def main():
    # create the training + test sets
    try:
        data = pd.read_csv('Data/BaseHackathon.csv')
        train = pd.read_csv('Data/Target_AgeGroup.csv')
    except IOError:
        print("io ERROR-->Could not locate file.")
    data.head(10)


if __name__ == '__main__':
    main()
    # import csv
    # r = csv.DictReader(open('Data/BaseHackathon.csv','r'))
    # a = {x: 0 for x in r.fieldnames}
    # b = 1
    # for row in r:
    #     b+= 1
    #     for k, v in row.items():
    #         if v == 'NA':
    #             a[k] += 1

