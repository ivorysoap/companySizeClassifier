from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import logistic
from tools import pickleFile
from tools import unpickleFile
import sklearn
import pandas as pd
import numpy as np
import datetime
import sys


def train_model(clf, X_train, y_train, epochs=10):
    """
    Trains a specific model and returns a list of results

    :param clf: sklearn model
    :param X_train: encoded training data (attributes)
    :param y_train: training data (attribute to predict
    :param epochs: number of iterations (default=10)
    :return: result (accuracy) for this training data
    """
    scores = []
    print("Starting training...")
    for i in range(1, epochs + 1):
        print("Epoch:" + str(i) + "/" + str(epochs) + " -- " + str(datetime.datetime.now()))
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        scores.append(score)
    print("Done training.  The score(s) is/are: " + str(scores))
    return scores

def cleanDataset(set):
    """ Custom method for cleaning our dataset. """
    print(len(set))
    print("of which we have this many NaNs:")
    print(set['year_founded'].isna().sum())

    pd.to_numeric(set['year_founded'], errors='coerce')

    set = set.dropna()

    print(set['size_range'].value_counts())

    print(len(set))
    print("of which we have this many NaNs:")
    print(set['year_founded'].isna().sum())

    return set

def peekPredictions(xTrain_sf_encoded, yTrain):

    print("= = = = = = = = = = = = = = = = = = = \n\nFirst 10 predictions:\n")

    yPredicted = lrModel.predict(xTrain_sf_encoded[0:10])
    print(yPredicted)
    print(yTrain[0:10])

    print(" = = = = = = = = = = = = = ")

    lrModel.predict_proba(xTrain_sf_encoded[0:10])

def getPrettyTimestamp():
    """ Returns timestamp: 'YYYY-MM-DD  HH:MM' """
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

def parseArgs():
    """ Parse program arguments. """
    if len(sys.argv) > 2 and sys.argv[1] == '--testonly':

        userRequestedTrain = False
        filename = str(sys.argv[2])

    elif len(sys.argv) == 1:

        userRequestedTrain = True
        filename = ''

    else:

        print("\nInvalid usage.  Usage:\n\n")
        print("\tpython modelTrainer.py [--testonly] [filename]\n")
        print("\nThe [--testonly] flag specifies to test an existing machine-learning model, instead of creating and "
              "training one.\nIf using the --testonly flag, you have to specify a [filename] from which to load the "
              "existing model.")
        sys.exit(1)

    return userRequestedTrain, filename


def main():

    # TODO Remove print()s throughout

    # Parse the arguments.
    userRequestedTrain, filename = parseArgs()

    # Some custom Pandas settings - TODO remove this
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_rows', 1000)

    dataset = pd.read_csv("companies_sorted.csv", nrows=10000)

    dataset = cleanDataset(dataset)

    # size_range is the attribute to be predicted, so we pop it from the dataset.  Execute only once!
    sizeRange = dataset.pop("size_range").values

    print(dataset[dataset.columns[1:4]].head(5))

    # We split our dataset and attribute-to-be-preditcted into training and testing subsets.
    xTrain, xTest, yTrain, yTest = train_test_split(dataset, sizeRange, test_size=0.25, random_state=1)

    le = LabelEncoder()
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    # Our feature set, i.e. the inputs to our machine-learning model.
    featureSet = ['name', 'year_founded']

    # Making a copy of test and train sets with only the columns we want.
    xTrain_sf = xTrain[featureSet].copy()
    xTest_sf = xTest[featureSet].copy()

    print("The following are the first 5 entries of xTrain_Sf")

    print(xTrain_sf.head(5))

    print("------")

    xTrain_sf_le = xTrain_sf.apply(le.fit_transform)

    print("The following are the first 15 entries of xTrain_sf_le")

    print(xTrain_sf_le.head(15))

    print("--------")

    # Apply one-hot encoding to columns
    ohe.fit(xTrain_sf)

    featureNames = ohe.get_feature_names()
    print(featureNames)

    # Encoding test and train sets together
    xTrain_sf_encoded = ohe.transform(xTrain_sf)
    xTest_sf_encoded = ohe.transform(xTest_sf)

    print(xTrain.shape)

    # ------ using Logistic Regression classifier - training phase ------

    if userRequestedTrain:

        # We define the model we're going to use.
        lrModel = LogisticRegression(solver='lbfgs', multi_class="multinomial", max_iter=1000, random_state=1)

        # Now, let's train it.
        lrScores = train_model(lrModel, xTrain_sf_encoded, yTrain, 1)

        print(lrScores)

        print("Training done! Pickling....")

        # Save the model as a file.
        filename = "models/Model_" + getPrettyTimestamp()
        pickleFile(lrModel, filename)

    # Reload the model for testing.  If we didn't train the model ourselves, then it was specified as an argument.
    lrModel = unpickleFile(filename)

    print(xTrain_sf_encoded[0:10])

    PRED = lrModel.predict(xTrain_sf_encoded[0:10])

    print("Unpickled successfully.")

    # ---- Doing a few predictions to get a rough idea of accuracy -----

    # peekPredictions(xTrain_sf_encoded, yTrain)

    # ----- testing phase -------

    testLrScores = train_model(lrModel, xTest_sf_encoded, yTest, 1)
    print(testLrScores)

    # trainScore = lrScores[0]
    trainScore = 0.9201578143173162
    testScore = testLrScores[0]

    scores = sorted([(trainScore, 'train'), (testScore, 'test')], key=lambda x: x[0], reverse=True)
    better_score = scores[0]  # largest score
    print(scores)

    # Which score was better?
    print("Better score: %s" % "{}".format(better_score))

    print("Pickling....")

    pickleFile(lrModel, "models/TESTING_" + getPrettyTimestamp())


if __name__ == "__main__":
    main()