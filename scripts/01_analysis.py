# Heavily influenced by: https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter?login=true#

import pandas as pd
import lightgbm as lgbm
import numpy as np
import os
import scripts.donorchoose_functions as fn
import re

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from datetime import datetime

from tqdm import tqdm

# Reading in data

dtype = {
    'id': str,
    'teacher_id': str,
    'teacher_prefix': str,
    'school_state': str,
    'project_submitted_datetime': str,
    'project_grade_category': str,
    'project_subject_categories': str,
    'project_subject_subcategories': str,
    'project_title': str,
    'project_essay_1': str,
    'project_essay_2': str,
    'project_essay_3': str,
    'project_essay_4': str,
    'project_resource_summary': str,
    'teacher_number_of_previously_posted_projects': int,
    'project_is_approved': np.uint8,
}

data_dir = "F:/Nerdy Stuff/Kaggle/DonorsChoose"
sub_path = "F:/Nerdy Stuff/Kaggle submissions/DonorChoose"

train = pd.read_csv(os.path.join(data_dir, "data/train.csv"), dtype=dtype)
test = pd.read_csv(os.path.join(data_dir, "data/test.csv"), dtype=dtype)

print("Extracting text features")

train = fn.extract_text_features(train)
test = fn.extract_text_features(test)

print("Extracting datetime features")

train = fn.extract_timestamp_features(train)
test = fn.extract_timestamp_features(test)

print("Joining together essays")

train['project_essay'] = fn.join_essays(train)
test['project_essay'] = fn.join_essays(test)

train = train.drop([
    'project_essay_1', 'project_essay_2',
     'project_essay_3', 'project_essay_4'
    ], axis=1)

test = test.drop([
    'project_essay_1', 'project_essay_2',
     'project_essay_3', 'project_essay_4'
    ], axis=1)

sample_sub = pd.read_csv(os.path.join("data/sample_submission.csv"))
res = pd.read_csv(os.path.join(data_dir, "data/resources.csv"))

id_test = test['id'].values

# Rolling up resources to one row per application

print("Rolling up resource requirements to one line and creating aggregate feats")

res = (res
        .groupby('id').apply(fn.price_quantity_agg)
        .reset_index())

res['mean_price'] = res['price_sum']/res['quantity_sum']

print("Train has %s rows and %s cols" % (train.shape[0], train.shape[1]))
print("Test has %s rows and %s cols" % (test.shape[0], test.shape[1]))
print("Res has %s rows and %s cols" % (res.shape[0], res.shape[1]))

print("Train has %s more rows than test" % (train.shape[0] / test.shape[0]))

train = pd.merge(left=train, right=res, on="id", how="left")
test = pd.merge(left=test, right=res, on="id", how="left")

print("Train after merge has %s rows and %s cols" % (train.shape[0], train.shape[1]))
print("Test after merge has %s rows and %s cols" % (test.shape[0], test.shape[1]))

print("Concatenating datasets so I can build the label encoders")

df_all = pd.concat([train, test], axis=0)

# TF-IDF and label encoding - will take first iteration from kaggle script and then move to sklearn pipelines?
# Renaming these cols to include later on

print('Label Encoder...')

col_rename = {
    'teacher_id': 'enc_teacher_id',
    'teacher_prefix': 'enc_teacher_prefix',
    'school_state': 'enc_school_state',
    'project_grade_category': 'enc_project_grade_category',
    'project_subject_categories': 'enc_project_subject_categories',
    'project_subject_subcategories': 'enc_project_subject_subcategories' # Can refactor to be more elegant

}

train = train.rename(columns=col_rename)
test = test.rename(columns=col_rename)
df_all = df_all.rename(columns=col_rename)

r = re.compile("enc_")

filtered = filter(r.match, train.columns)
cols = [i for i in filtered]

for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
del le

print("Modelling")

cols = train.columns

variables_names_to_include = ['price', 'quantity', '_wc',
                              '_len', 'subtime_', 'enc_']
vars_to_include = []

for variable in variables_names_to_include:

    regex = ".*" + variable + "*."
    print(regex)
    r = re.compile(regex)

    filtered = filter(r.match, cols)
    result = [i for i in filtered]

    for res in result:
        vars_to_include.append(res)

X_tr = train[vars_to_include]
y_tr = train['project_is_approved'].values
X_tst = test[vars_to_include]

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

print("Completed outputting to folder")
