import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 30)
pd.set_option('display.max_rows', 1000)

dataset = pd.read_csv("companies_sorted.csv", nrows=2000000)

# print(dataset['year_founded'].isna().sum())

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



def oneHotEncoding(name):
    """ Get a one-hot encoding for a given name (in this case, a company name) """
    encoding = np.zeros(150)
    for i in range(len(name)):
        encoding[i] = ord(name[i])
    return encoding
