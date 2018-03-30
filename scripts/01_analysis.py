import pandas as pd
import lightgbm as lgbm
import numpy as np
import os

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

# Reading in data

data_dir = "F:/Nerdy Stuff/Kaggle/DonorsChoose"
sub_path = "F:/Nerdy Stuff/Kaggle submissions/DonorChoose"

train = pd.read_csv(os.path.join(data_dir, "data/train.csv"))
res = pd.read_csv(os.path.join(data_dir, "data/resources.csv"))
test = pd.read_csv(os.path.join(data_dir, "data/test.csv"))
sample_sub = pd.read_csv(os.path.join("data/sample_submission.csv"))

print("Train has %s rows and %s cols" % (train.shape[0], train.shape[1]))
print("Test has %s rows and %s cols" % (test.shape[0], test.shape[1]))
print("Res has %s rows and %s cols" % (res.shape[0], res.shape[1]))

print("Train has %s more rows than test" % (train.shape[0] / test.shape[0]))

train = pd.merge(left=train, right=res, on="id", how="left")
test = pd.merge(left=test, right=res, on="id", how="left")

print("Train after merge has %s rows and %s cols" % (train.shape[0], train.shape[1]))
print("Test after merge has %s rows and %s cols" % (test.shape[0], test.shape[1]))

# Removing duplicates - shouldn't need this

id_test = test['id'].unique()

train['item_cost'] = train['price'] * train['quantity']
test['item_cost'] = test['price'] * test['quantity']

def my_agg(x):

    names = {
        'total_cost': x['item_cost'].sum(),
        'total_qty': x['quantity'].sum(),
        'num_items': x['quantity'].count(),
        'avg_price': x['price'].mean()

    }

    return pd.Series(names, index=['total_cost', 'total_qty',
                                   'num_items', 'avg_price'])


train_group = train.groupby(['id', 'project_is_approved']).apply(my_agg)
test_group = test.groupby('id').apply(my_agg)

train = train_group.reset_index()
test = test_group.reset_index()

print("Train after dropping dupes has %s rows and %s cols" % (train.shape[0], train.shape[1]))
print("Test after dropping dupes has %s rows and %s cols" % (test.shape[0], test.shape[1]))

# First iteration of modelling

X_tr = train.drop(['id', 'project_is_approved'], axis=1)
y_tr = train['project_is_approved'].values
X_tst = test.drop('id', axis=1)

fold_scores = []
skf = StratifiedKFold(n_splits=10)

clf = lgbm.LGBMClassifier()

for i, (train_idx, valid_idx) in enumerate(skf.split(X_tr, y_tr)):

    print("Fold #%s" % (i + 1))

    X_train, X_valid = X_tr.iloc[train_idx, :], X_tr.iloc[valid_idx, :]
    y_train, y_valid = y_tr[train_idx], y_tr[valid_idx]

    clf.fit(X_train, y_train)
    y_valid_predictions = clf.predict_proba(X_valid)[:, 1]

    auc_roc_score = roc_auc_score(y_valid, y_valid_predictions)
    fold_scores.append(auc_roc_score)

mean_score = round(np.mean(fold_scores), 3)
std_score = round(np.std(fold_scores), 3)

print('AUC = {:.3f} +/- {:.3f}'.format(mean_score, std_score))

# Fitting model on whole train

clf.fit(X_tr, y_tr)
predictions = clf.predict_proba(X_tst)[:, 1]

# Submitting to F:/

pred_set = pd.DataFrame()
pred_set['id'] = id_test
pred_set['project_is_approved'] = predictions

sub = sample_sub.drop('project_is_approved', axis=1)
sub = pd.merge(sub, pred_set, on='id', how='left')

now = datetime.strftime(datetime.now(), "%d_%m_%y_%H%M")
file_name_list = [now, str(mean_score), str(std_score)]
file_name = '_'.join(file_name_list) + ".csv"

sub.to_csv(os.path.join(sub_path, file_name), index=False)

