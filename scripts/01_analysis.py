import pandas as pd
from os import chdir

##################### Reading in data #####################

data_dir = "F:/Nerdy Stuff/Kaggle/DonorsChoose"

chdir(data_dir)

train = pd.read_csv("data/train.csv")
res = pd.read_csv("data/resources.csv")
test = pd.read_csv("data/test.csv")

print("Train has %s rows and %s cols" % (train.shape[0], train.shape[1]))
print("Test has %s rows and %s cols" % (test.shape[0], test.shape[1]))
print("Res has %s rows and %s cols" % (res.shape[0], res.shape[1]))

print("Train has %s more rows than test" % (train.shape[0] / test.shape[0]))

train = pd.merge(left=train, right=res, on="id")
test = pd.merge(left=test, right=res, on="id")

print("Train after merge has %s rows and %s cols" % (train.shape[0], train.shape[1]))
print("Test after merge has %s rows and %s cols" % (test.shape[0], test.shape[1]))

