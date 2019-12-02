import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = pd.read_csv("companies_sorted.csv")

print(dataset.head(15))

print(list(dataset))

print(dataset.shape)
