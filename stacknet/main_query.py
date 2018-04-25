import pylab as pl # linear algebra + plots
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import gc
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import roc_auc_score as auc
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict, Counter
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from scipy.stats import pearsonr
from scipy.sparse import hstack

import stacknet.stacknet_funcs as funcs
import os

Folder = "C:/Users/Evan/PycharmProjects/DonorChoose/stacknet/input/"
os.chdir(Folder)

Ttr = pd.read_csv('train.csv')
Tts = pd.read_csv('test.csv', low_memory=False)
R = pd.read_csv('resources.csv')

# Filtering rows for test

# Ttr = Ttr.iloc[0:1000, :]
# Tts = Tts.iloc[0:1000, :]

# combine the tables into one
target = 'project_is_approved'
Ttr['tr'] = 1; Tts['tr'] = 0
Ttr['ts'] = 0; Tts['ts'] = 1

T = pd.concat((Ttr,Tts))

T.loc[T.project_essay_4.isnull(), ['project_essay_4','project_essay_2']] = \
    T.loc[T.project_essay_4.isnull(), ['project_essay_2','project_essay_4']].values

T[['project_essay_2','project_essay_3']] = T[['project_essay_2','project_essay_3']].fillna('')

T['project_essay_1'] = T.apply(lambda row: ' '.join([str(row['project_essay_1']),
                                                     str(row['project_essay_2'])]), axis=1)
T['project_essay_2'] = T.apply(lambda row: ' '.join([str(row['project_essay_3']),
                                                     str(row['project_essay_4'])]), axis=1)

T = T.drop(['project_essay_3', 'project_essay_4'], axis=1)

R['priceAll'] = R['quantity']*R['price']
newR = R.groupby('id').agg({'description':'count',
                            'quantity':'sum',
                            'price':'sum',
                            'priceAll':'sum'}).rename(columns={'description':'items'})
newR['avgPrice'] = newR.priceAll / newR.quantity
numFeatures = ['items', 'quantity', 'price', 'priceAll', 'avgPrice']

for func in ['min', 'max', 'mean']:
    newR = newR.join(R.groupby('id').agg({'quantity':func,
                                          'price':func,
                                          'priceAll':func}).rename(
                                columns={'quantity':func+'Quantity',
                                         'price':func+'Price',
                                         'priceAll':func+'PriceAll'}).fillna(0))
    numFeatures += [func+'Quantity', func+'Price', func+'PriceAll']

newR = newR.join(R.groupby('id').agg({'description':lambda x:' '.join(x.values.astype(str))}).rename(
    columns={'description':'resource_description'}))

T = T.join(newR, on='id')

# if you visit the donors website, it has categorized the price by these bins:
T['price_category'] = pl.digitize(T.priceAll, [0, 50, 100, 250, 500, 1000, pl.inf])
numFeatures.append('price_category')
# the difference of max and min of price and quantity per item can also be relevant
for c in ['Quantity', 'Price', 'PriceAll']:
    T['max%s_min%s'%(c,c)] = T['max%s'%c] - T['min%s'%c]
    numFeatures.append('max%s_min%s'%(c,c))

del Ttr, Tts, R, newR
gc.collect();

le = LabelEncoder()
T['teacher_id'] = le.fit_transform(T['teacher_id'])
T['teacher_gender_unknown'] = T.teacher_prefix.apply(lambda x:int(x not in ['Ms.', 'Mrs.', 'Mr.']))
numFeatures += ['teacher_number_of_previously_posted_projects','teacher_id','teacher_gender_unknown']

statFeatures = []
for col in ['school_state', 'teacher_id', 'teacher_prefix', 'teacher_gender_unknown', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories', 'teacher_number_of_previously_posted_projects']:
    Stat = T[['id', col]].groupby(col).agg('count').rename(columns={'id':col+'_stat'})
    Stat /= Stat.sum()
    T = T.join(Stat, on=col)
    statFeatures.append(col+'_stat')

textColumns = ['project_essay_1', 'project_essay_2', 'project_resource_summary', 'resource_description', 'project_title']

print('key words')
KeyChars = ['!', '\?', '@', '#', '\$', '%', '&', '\*', '\(', '\[', '\{', '\|', '-', '_', '=', '\+',
            '\.', ':', ';', ',', '/', '\\\\r', '\\\\t', '\\"', '\.\.\.', 'etc', 'http',
            'military', 'traditional', 'charter', 'head start', 'magnet', 'year-round', 'alternative',
            'art', 'book', 'basics', 'computer', 'laptop', 'tablet', 'kit', 'game', 'seat',
            'food', 'cloth', 'hygiene', 'instraction', 'technolog', 'lab', 'equipment',
            'music', 'instrument', 'nook', 'desk', 'storage', 'sport', 'exercise', 'trip', 'visitor',
            'my students', 'our students', 'my class', 'our class']
for col in textColumns:
    for c in KeyChars:
        T[col+'_'+c] = T[col].apply(lambda x: len(re.findall(c, x.lower())))
        numFeatures.append(col+'_'+c)

#####
print('num words')
for col in textColumns:
    T['n_'+col] = T[col].apply(lambda x: len(x.split()))
    numFeatures.append('n_'+col)
    T['nUpper_'+col] = T[col].apply(lambda x: sum([s.isupper() for s in list(x)]))
    numFeatures.append('nUpper_'+col)

#####
print('word tags')
Tags = ['CC', 'CD', 'DT', 'IN', 'JJ', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
        'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO',
        'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

#####
print('common words')
for i, col1 in enumerate(textColumns[:-1]):
    for col2 in textColumns[i+1:]:
        T['%s_%s_common' % (col1, col2)] = T.apply(lambda row:len(set(re.split('\W', row[col1].lower())).intersection(re.split('\W', row[col2].lower()))), axis=1)
        numFeatures.append('%s_%s_common' % (col1, col2))

dateCol = 'project_submitted_datetime'

T[dateCol] = pd.to_datetime(T[dateCol])
T = funcs.getTimeFeatures(T, dateCol)

P_tar = T[T.tr==1][target].mean()
timeFeatures = ['year', 'month', 'day', 'dow', 'hour', 'days']
for col in timeFeatures:
    Stat = T[['id', col]].groupby(col).agg('count').rename(columns={'id':col+'_stat'})
    Stat /= Stat.sum()
    T = T.join(Stat, on=col)
    statFeatures.append(col+'_stat')

numFeatures += timeFeatures
numFeatures += statFeatures

T2 = T[numFeatures+['id','tr','ts',target]].copy()
Ttr = T2[T.tr==1]
Tar_tr = Ttr[target].values
n = 10
inx = [pl.randint(0, Ttr.shape[0], int(Ttr.shape[0]/n)) for k in range(n)]
# inx is used for crossvalidation of calculating the correlation and p-value
Corr = {}
for c in numFeatures:
    # since some values might be 0s, I use x+1 to avoid missing some important relations
    C1,P1=pl.nanmean([pearsonr(Tar_tr[inx[k]],   (1+Ttr[c].iloc[inx[k]])) for k in range(n)], 0)
    C2,P2=pl.nanmean([pearsonr(Tar_tr[inx[k]], 1/(1+Ttr[c].iloc[inx[k]])) for k in range(n)], 0)
    if P2<P1:
        T2[c] = 1/(1+T2[c])
        Corr[c] = [C2,P2]
    else:
        T2[c] = 1+T2[c]
        Corr[c] = [C1,P1]

polyCol = []
thrP = 0.01
thrC = 0.02
print('columns \t\t\t Corr1 \t\t Corr2 \t\t Corr Combined')
for i, c1 in enumerate(numFeatures[:-1]):
    C1, P1 = Corr[c1]
    for c2 in numFeatures[i+1:]:
        C2, P2 = Corr[c2]
        V = T2[c1] * T2[c2]
        Vtr = V[T2.tr==1].values
        C, P = pl.nanmean([pearsonr(Tar_tr[inx[k]], Vtr[inx[k]]) for k in range(n)], 0)
        if P<thrP and abs(C) - max(abs(C1),abs(C2)) > thrC:
            T[c1+'_'+c2+'_poly'] = V
            polyCol.append(c1+'_'+c2+'_poly')
            print(c1+'_'+c2, '\t\t(%g, %g)\t(%g, %g)\t(%g, %g)'%(C1,P1, C2,P2, C,P))

numFeatures += polyCol
print(len(numFeatures))
del T2, Ttr
gc.collect();

X_tp = funcs.getCatFeatures(T, 'teacher_prefix')
X_ss = funcs.getCatFeatures(T, 'school_state')
X_pgc = funcs.getCatFeatures(T, 'project_grade_category')
X_psc = funcs.getCatFeatures(T, 'project_subject_categories')
X_pssc = funcs.getCatFeatures(T, 'project_subject_subcategories')

X_cat = hstack((X_tp, X_ss, X_pgc, X_psc, X_pssc))

del X_tp, X_ss, X_pgc, X_psc, X_pssc

p = PorterStemmer()
def wordPreProcess(sentence):
    return ' '.join([p.stem(x.lower()) for x in re.split('\W', sentence) if len(x) >= 1])

def getTextFeatures(T, Col, max_features=10000, verbose=True):
    if verbose:
        print('processing: ', Col)
    vectorizer = CountVectorizer(stop_words=None,
                                 preprocessor=wordPreProcess,
                                 max_features=max_features,
                                 binary=True,
                                 ngram_range=(1,2))
#     vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'),
#                                  preprocessor=wordPreProcess,
#                                  max_features=max_features)
    X = vectorizer.fit_transform(T[Col])
    return X, vectorizer.get_feature_names()

n_es1, n_es2, n_prs, n_rd, n_pt = 3000, 8000, 2000, 3000, 1000
X_es1, feat_es1 = getTextFeatures(T, 'project_essay_1', max_features=n_es1)
X_es2, feat_es2 = getTextFeatures(T, 'project_essay_2', max_features=n_es2)
X_prs, feat_prs = getTextFeatures(T, 'project_resource_summary', max_features=n_prs)
X_rd, feat_rd = getTextFeatures(T, 'resource_description', max_features=n_rd)
X_pt, feat_pt = getTextFeatures(T, 'project_title', max_features=n_pt)

X_txt = hstack((X_es1, X_es2, X_prs, X_rd, X_pt))
del X_es1, X_es2, X_prs, X_rd, X_pt

from sklearn.preprocessing import StandardScaler
X = hstack((X_txt, X_cat, StandardScaler().fit_transform(T[numFeatures].fillna(0)))).tocsr()

Xtr = X[pl.find(T.tr==1), :]
Xts = X[pl.find(T.ts==1), :]
Ttr_tar = T[T.tr==1][target].values
Tts = T[T.ts==1][['id',target]]

Yts = []
del T
del X
gc.collect();

folder = 'C:/Users/Evan/PycharmProjects/DonorChoose/stacknet'
os.chdir(folder)

funcs.from_sparse_to_file("train.sparse", Xtr, deli1=" ", deli2=":", ytarget=Ttr_tar)
funcs.from_sparse_to_file("test.sparse", Xts, deli1=" ", deli2=":", ytarget=None)