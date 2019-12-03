import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import datetime
import pandas as pd
import numpy as np
import datetime
import pickle
import sys

def oneHotEncoding(name):
    """ Get a one-hot encoding for a given name (in this case, a company name) """
    encoding = np.zeros(150)
    for i in range(len(name)):
        encoding[i] = ord(name[i])
    return encoding


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

def _pickle(data, filename):
    pickle.dump(data, open(filename, 'wb'))
    return

def _unpickle(filename):
    return pickle.load(open(filename, 'rb'))

def getPrettyTimestamp():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 1000)

dataset = pd.read_csv("companies_sorted.csv", nrows=100000)

print(len(dataset))
print("of which we have this many NaNs:")
print(dataset['year_founded'].isna().sum())

pd.to_numeric(dataset['year_founded'], errors='coerce')

# dataset = dataset.dropna(subset=["year_founded"])  # Remove NaN values from year_founded column

dataset = dataset.dropna()

print(len(dataset))
print("of which we have this many NaNs:")
print(dataset['year_founded'].isna().sum())

sizeRange = dataset.pop("size_range").values

print(dataset[dataset.columns[1:4]].head(5))

xTrain, xTest, yTrain, yTest = train_test_split(dataset, sizeRange, test_size=0.25, random_state=1)

le = LabelEncoder()

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

featureSet = ['name', 'year_founded']

# Making a copy of test and train datasets with only the columns we want
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

if len(sys.argv) > 1:

    # We define the model we're going to use
    lrModel = LogisticRegression(solver='lbfgs', multi_class="multinomial", max_iter=1000, random_state=1)

    # Now, let's train it
    lrScores = train_model(lrModel, xTrain_sf_encoded, yTrain, 1)

    print(lrScores)

    print("Pickling....")

    _pickle(lrModel, "models/Model_" + getPrettyTimestamp())

lrModel = _unpickle("models/Model_2019-12-03 12:08")


# ---- Doing a few predictions to get a rough idea of accuracy -----

print("= = = = = = = = = = = = = = = = = = = \n\nFirst 10 predictions:\n")

yPredicted = lrModel.predict(xTrain_sf_encoded[0:10])
print(yPredicted)
print(yTrain[0:10])

print (" = = = = = = = = = = = = = ")

lrModel.predict_proba(xTrain_sf_encoded[0:10])



# ----- testing phase -------



