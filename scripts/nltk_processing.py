# Heavily influenced by: https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter?login=true#

import pandas as pd
import numpy as np
import os
import scripts.donorchoose_functions as fn
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

# Creating vars that determine what will be run and what won't

train = pd.read_csv(os.path.join(data_dir, "data/train.csv"),
                    dtype=dtype, parse_dates=['project_submitted_datetime'])
test = pd.read_csv(os.path.join(data_dir, "data/test.csv"),
                   dtype=dtype, parse_dates=['project_submitted_datetime'])

train['project_essay'] = fn.join_essays(train)
test['project_essay'] = fn.join_essays(test)

print("Rolling up resource requirements to one line and creating aggregate feats")

res = pd.read_csv(os.path.join(data_dir, "data/resources.csv"))

res = (res
        .groupby('id').apply(fn.price_quantity_agg)
        .reset_index())

res['mean_price'] = res['price_sum'] / res['quantity_sum']
res['price_max_to_price_min'] = res['price_max']/res['price_min']
res['quantity_max_to_quantity_min'] = res['quantity_max']/res['quantity_min']

train = pd.merge(left=train, right=res, on="id", how="left")
test = pd.merge(left=test, right=res, on="id", how="left")

print("Recoding missing values in teacher_prefix")

train['teacher_prefix'] = train['teacher_prefix'].fillna('Unknown')
test['teacher_prefix'] = test['teacher_prefix'].fillna('Unknown')

text_labels = ['teacher_id', 'teacher_prefix', 'school_state',
               'project_grade_category', 'project_subject_categories',
               'project_subject_subcategories', 'description_ttl']

print("Concatenating datasets so I can build the label encoders")

for c in tqdm(text_labels):

    train[c] = train[c].astype(str)
    test[c] = test[c].astype(str)

essay_cols_nlp = ['project_title', 'project_essay',
                  'project_resource_summary']

n_features = [400,4000,400, 400]

def remove_stops_punct_stemmer(essay):

    """

    :param essay: Essay for stemming and removing stop words
    :return: A cleaned and stemmed essay
    """

    from nltk.corpus import stopwords
    from string import punctuation
    from nltk.stem import WordNetLemmatizer

    wordnet_lemmatizer = WordNetLemmatizer()

    stopwords_en = set(stopwords.words('english'))
    stopwords_en_withpunct = stopwords_en.union(set(punctuation))
    stopwords_en_withpunct.update(['title', 'come'])

    essay = essay.lower()
    words = filter(None, essay.split(" "))

    words = [word for word in words if word not in stopwords_en_withpunct]
    words = [w for w in words if not w.isdigit()] #Removing digits

    words = [wordnet_lemmatizer.lemmatize(w) for w in words]

    result = ' '.join(words)

    return result

print("Stemming / NLP cols")

for c in tqdm(essay_cols_nlp):

    train[c] = train[c].apply(lambda x: remove_stops_punct_stemmer(x))
    test[c] = test[c].apply(lambda x: remove_stops_punct_stemmer(x))

    print("Outputting stemmed and stopword removed essays to save time")

train.to_csv(os.path.join(data_dir, 'data/train_stem.csv'))
test.to_csv(os.path.join(data_dir, 'data/test_stem.csv'))

print("Train/test stem outputted")
