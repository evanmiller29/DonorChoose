import gc
import numpy as np
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import LabelEncoder

from datetime import datetime

from tqdm import tqdm
import lightgbm as lgb

# Load Data
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

train = pd.read_csv(os.path.join(data_dir, 'data/train_stem.csv'),
                    low_memory=True)
test = pd.read_csv(os.path.join(data_dir, 'data/test_stem.csv'),
                   low_memory=True)

id_test = test['id'].values

# Extract features
def extract_features(df):
    df['project_title_len'] = df['project_title'].apply(lambda x: len(str(x)))
    df['project_essay_1_len'] = df['project_essay_1'].apply(lambda x: len(str(x)))
    df['project_essay_2_len'] = df['project_essay_2'].apply(lambda x: len(str(x)))
    df['project_essay_3_len'] = df['project_essay_3'].apply(lambda x: len(str(x)))
    df['project_essay_4_len'] = df['project_essay_4'].apply(lambda x: len(str(x)))
    df['project_resource_summary_len'] = df['project_resource_summary'].apply(lambda x: len(str(x)))

    df['project_title_wc'] = df['project_title'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_1_wc'] = df['project_essay_1'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_2_wc'] = df['project_essay_2'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_3_wc'] = df['project_essay_3'].apply(lambda x: len(str(x).split(' ')))
    df['project_essay_4_wc'] = df['project_essay_4'].apply(lambda x: len(str(x).split(' ')))
    df['project_resource_summary_wc'] = df['project_resource_summary'].apply(lambda x: len(str(x).split(' ')))

extract_features(train)
extract_features(test)

train.drop([
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4'], axis=1, inplace=True)
test.drop([
    'project_essay_1',
    'project_essay_2',
    'project_essay_3',
    'project_essay_4'], axis=1, inplace=True)

# Recoding as when stopwords are removed some titles have no values

print("Recoding missing values once NLP preprocessing done. Might want to check that")

train.loc[train['project_title'].isnull() == True, 'project_title'] = 'No values once NLP preprocessing is done'
test.loc[test['project_title'].isnull() == True, 'project_title'] = 'No values once NLP preprocessing is done'

train.loc[train['project_essay'].isnull() == True, 'project_essay'] = 'No values once NLP preprocessing is done'
test.loc[test['project_essay'].isnull() == True, 'project_essay'] = 'No values once NLP preprocessing is done'

train.loc[train['project_resource_summary'].isnull() == True, 'project_resource_summary'] = 'No values once NLP preprocessing is done'
test.loc[test['project_resource_summary'].isnull() == True, 'project_resource_summary'] = 'No values once NLP preprocessing is done'

train.loc[train['description_ttl'].isnull() == True, 'description_ttl'] = 'No values once NLP preprocessing is done'
test.loc[test['description_ttl'].isnull() == True, 'description_ttl'] = 'No values once NLP preprocessing is done'

gc.collect()

# Preprocess columns with label encoder
print('Label Encoder...')
cols = [
    'teacher_id',
    'teacher_prefix',
    'school_state',
    'project_grade_category',
    'project_subject_categories',
    'project_subject_subcategories'
]

df_all = pd.concat([train, test], axis=0)

for c in tqdm(cols):
    le = LabelEncoder()
    le.fit(df_all[c].astype(str))
    train[c] = le.transform(train[c].astype(str))
    test[c] = le.transform(test[c].astype(str))
del le
gc.collect()
print('Done.')

# Preprocess timestamp
print('Preprocessing timestamp...')

def process_timestamp(df):

    df['project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime'])

    df['year'] = df['project_submitted_datetime'].apply(lambda x: x.year)
    df['month'] = df['project_submitted_datetime'].apply(lambda x: x.month)
    df['day'] = df['project_submitted_datetime'].apply(lambda x: x.day)
    df['day_of_week'] = df['project_submitted_datetime'].apply(lambda x: x.dayofweek)

    df['hour'] = df['project_submitted_datetime'].apply(lambda x: x.hour)
    df['minute'] = df['project_submitted_datetime'].apply(lambda x: x.minute)

    df['project_submitted_datetime'] = df['project_submitted_datetime'].values.astype(np.int64)

process_timestamp(train)
process_timestamp(test)
print('Done.')

# Preprocess text
print('Preprocessing text...')

cols = [
    'project_title',
    'project_essay',
    'project_resource_summary',
    'description_ttl'
]
n_features = [
    400,
    4040,
    400,
    400
]

for c_i, c in tqdm(enumerate(cols)):

    print("TFIDF for %s" % (c))

    tfidf = TfidfVectorizer(
        max_features=n_features[c_i],
        norm='l2',
    )
    tfidf.fit(df_all[c])
    tfidf_train = np.array(tfidf.transform(train[c]).toarray(), dtype=np.float16)
    tfidf_test = np.array(tfidf.transform(test[c]).toarray(), dtype=np.float16)

    for i in range(n_features[c_i]):
        train[c + '_tfidf_' + str(i)] = tfidf_train[:, i]
        test[c + '_tfidf_' + str(i)] = tfidf_test[:, i]

    del tfidf, tfidf_train, tfidf_test
    gc.collect()

print('Done.')
gc.collect()

# Prepare data
cols_to_drop = [
        'Unnamed: 0'
    ,   'id'
    ,   'teacher_id'
    ,   'project_title'
    ,   'project_essay'
    ,   'project_resource_summary'
    ,   'project_is_approved'
    ,   'description_ttl'
]

X = train.drop(cols_to_drop, axis=1, errors='ignore')
y = train['project_is_approved']
X_test = test.drop(cols_to_drop, axis=1, errors='ignore')
id_test = test['id'].values
feature_names = list(X.columns)
print(X.shape, X_test.shape)

# del train, test
gc.collect()

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=0)
auc_buf = []

num_rows = 60000

X_train_test = X.iloc[0:num_rows, :]
y_train_test = y.iloc[0:num_rows]

prob_ests = []
y_test = []

prb = np.array(prob_ests[0])
y_tst = np.asarray(y_test[0], np.int32)

prb.dtype
y_tst.dtype

prb.shape
y_tst.shape

prb_ser = pd.Series(prb)
roc_auc_score(np.asarray(y_tst[0:9000], np.int32), prb[0:9000])

import matplotlib.pyplot as plt

pd.Series(prb[0:9000]).dtype

for train_index, valid_index in kf.split(X_train_test):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 14,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0,
    }

    lgb_train = lgb.Dataset(
        X_train_test.loc[train_index],
        y_train_test.loc[train_index],
        feature_name=feature_names,
    )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X_train_test.loc[valid_index],
        y_train_test.loc[valid_index],
    )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        # num_boost_round=10000,
        num_boost_round=100,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(60):
            if i < len(tuples):
                print(tuples[i])
            else:
                break

        del importance, model_fnames, tuples

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)

    print(type(p))
    print(p[0:5])
    print(type(X))
    print(type(y))
    print(max(p))

    prob_ests.append(p)
    y_test.append(y.loc[valid_index])
    auc = roc_auc_score(y.loc[valid_index], p)
    auc = round(auc, 4)

    print('{} AUC: {}'.format(str(cnt), str(auc)))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    auc_buf.append(auc)

    cnt += 1
    if cnt > 0:  # Comment this to run several folds
        break

    del model, lgb_train, lgb_valid, p
    gc.collect

auc_mean = round(np.mean(auc_buf), 3)
auc_std = round(np.std(auc_buf), 3)
print('AUC = {:.3f} +/- {:.3f}'.format(auc_mean, auc_std))

preds = p_buf / cnt

# Prepare submission
sub = pd.DataFrame()
sub['id'] = id_test
sub['project_is_approved'] = preds

now = datetime.strftime(datetime.now(), "%d_%m_%y_%H%M")
file_name_list = [now, str(auc_mean), str(auc_std)]
file_name = '_'.join(file_name_list) + ".csv"

sub.to_csv(os.path.join(sub_path, file_name), index=False)