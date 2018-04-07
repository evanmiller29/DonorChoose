# Heavily influenced by: https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter?login=true#

import pandas as pd
import lightgbm as lgbm
import numpy as np
import os
import scripts.donorchoose_functions as fn

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn_pandas import DataFrameMapper

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

# Write code that limits the rows until I've sorted out the kinks

data_dir = "F:/Nerdy Stuff/Kaggle/DonorsChoose"
sub_path = "F:/Nerdy Stuff/Kaggle submissions/DonorChoose"

train = pd.read_csv(os.path.join(data_dir, "data/train.csv"),
                    dtype=dtype, parse_dates=['project_submitted_datetime'])
test = pd.read_csv(os.path.join(data_dir, "data/test.csv"),
                   dtype=dtype, parse_dates=['project_submitted_datetime'])

train = train.iloc[0:100, :]
test= test.iloc[0:100, :]

train['project_essay'] = fn.join_essays(train)
test['project_essay'] = fn.join_essays(test)

sample_sub = pd.read_csv(os.path.join("data/sample_submission.csv"))
res = pd.read_csv(os.path.join(data_dir, "data/resources.csv"))

id_test = test['id'].values

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

print("Extracting text features")
print("Extracting datetime features")

print("Recoding missing values in teacher_prefix")

train['teacher_prefix'] = train['teacher_prefix'].fillna('Unknown')
test['teacher_prefix'] = test['teacher_prefix'].fillna('Unknown')

test['project_is_approved'] = 0

essay_cols = ['project_essay_1', 'project_essay_2', 'project_essay_3',
              'project_essay_4', 'project_resource_summary']

text_labels = ['teacher_id', 'teacher_prefix',
               'school_state','project_grade_category',
               'project_subject_categories','project_subject_subcategories']

transform_cols = ['project_submitted_datetime'] + essay_cols + text_labels

print("Concatenating datasets so I can build the label encoders")

df_all = pd.concat([train, test], axis=0)

for c in tqdm(text_labels):

    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)

print("Doing TFIDF")

essay_cols_nlp = ['project_title', 'project_essay', 'project_resource_summary', 'description']
n_features = [400,4040,400, 40]

for c_i, c in tqdm(enumerate(essay_cols_nlp)):
    tfidf = TfidfVectorizer(max_features=n_features[c_i],norm='l2',)

    tfidf.fit(df_all[c])
    tfidf_train = np.array(tfidf.transform(train[c]).toarray(), dtype=np.float16)
    tfidf_test = np.array(tfidf.transform(test[c]).toarray(), dtype=np.float16)

    for i in range(n_features[c_i]):
        train[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
        test[c + '_tfidf_' + str(i)] = tfidf_test[:, i]

print("Dropping TF-IDF cols after extracting information..")

train = train.drop(essay_cols_nlp, axis=1)
test = test.drop(essay_cols_nlp, axis=1)

print("Saving id and then dropping")

test_id = test['id']

y_train = train['project_is_approved'].values
X_train = train.drop('project_is_approved', axis=1)

X_train = X_train.drop('id', axis=1)
test = test.drop('id', axis=1)

columns = X_train.columns
no_transform_cols = [col for col in columns if col not in transform_cols]

print("Building and using pipeline")

feature_engineering_mapper = DataFrameMapper([

     ('project_submitted_datetime', fn.DateEncoder()),
     ('project_essay_1', fn.TextSummaryStatEncoder()),
     ('project_essay_2', fn.TextSummaryStatEncoder()),
     ('project_essay_3', fn.TextSummaryStatEncoder()),
     ('project_essay_4', fn.TextSummaryStatEncoder()),
     ('teacher_id', LabelEncoder()),
     ('teacher_prefix', LabelEncoder()),
     ('school_state', LabelEncoder()),
     ('project_grade_category', LabelEncoder()),
     ('project_subject_categories', LabelEncoder()),
     ('project_subject_subcategories', LabelEncoder()),
     (no_transform_cols, None)
     ], input_df=True)

feature_engineering_mapper.fit(df_all)

X_train = feature_engineering_mapper.transform(X_train)
test = feature_engineering_mapper.transform(test)

fold_scores = []
skf = StratifiedKFold(n_splits=10, random_state=1234)

clf = lgbm.LGBMClassifier()

for i, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):

    print("Fold #%s" % (i + 1))

    X_tr, X_valid = X_train[train_idx, :], X_train[valid_idx, :]
    y_tr, y_valid = y_train[train_idx], y_train[valid_idx]

    clf.fit(X_tr, y_tr)
    y_valid_predictions = clf.predict_proba(X_valid)[:, 1]

    auc_roc_score = roc_auc_score(y_valid, y_valid_predictions)
    fold_scores.append(auc_roc_score)

mean_score = round(np.mean(fold_scores), 3)
std_score = round(np.std(fold_scores), 3)

print('AUC = {:.3f} +/- {:.3f}'.format(mean_score, std_score))

clf.fit(X_tr, y_tr)
predictions = clf.predict_proba(X_test)[:, 1]

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
