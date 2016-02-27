import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    # create the training + test sets
    try:
        data = pd.read_csv('Data/BaseHackathon.csv')
    except IOError:
        print("io ERROR-->Could not locate file.")

    target = data.Activity.values

    train = data.drop('Activity', axis = 1).values

    test = pd.read_csv('Data/test.csv').values

    # create and train the random forest and call it 'rf'
    # --> n_estimators = the number of trees in this forest, viz.
    #     100 trees of forest
    # --> n_jobs set to -1 will use the number of cores present on your system.
    rf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    # fit(X, y[, sample_weight]) = build a forest of tress from the
    # training set (X, y)
    rf.fit(train, target)

    # predict_proba(X) predict class probabilities for X as list
    predicted_probs = [x[1] for x in rf.predict_proba(test)]

    # prep data for use in pd.Series
    molID, predictProbs = prepData(predicted_probs)

    # use a dictionary with keys as col headers and values as lists pulled from
    # previous prep function
    df = {'MoleculeID': molID, 'PredictedProbability': predictProbs}


    # pandas DataFrame = a tabular datastructure like a SQL table
    predicted_probs = pd.DataFrame(df)

    # write predicted_probs to file with pandas method .to_csv()--add header
    # for submission
    try:
        predicted_probs.to_csv('Data/submission.csv', index = False)
        print("File successfully written; check 'Data' folder")
    except IOError:
        print("io ERROR-->Could not write data to file.")

# preparing data for conversion to pd.DataFrame
def prepData(alist):
        # prepare list to be converted to pandas Series
        colOne = []
        colTwo = []
        idx = 1

        # for loop to set MoleculeID to match the benchmark;
        # place values into list for easier wrangling as pd.Series
        for i in alist:
            colOne.append(idx)
            colTwo.append(i)
            idx += 1

        return colOne, colTwo

# call the main function
main()