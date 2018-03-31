from sklearn.base import TransformerMixin
import pandas as pd

class DateEncoder(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        dt = X.dt
        return pd.concat([dt.year, dt.month, dt.day,
                          dt.quarter, dt.weekday,
                          dt.hour, dt.minute], axis=1)


dates_df = pd.DataFrame(
         {'dates': pd.date_range('2015-10-30 10:30', '2015-11-02 10:30'),
          'index': [1, 2, 3, 4]})

 mapper_dates = DataFrameMapper([
     ('dates', DateEncoder()),
     ('index', None)
     ], input_df=True)
print(mapper_dates.fit_transform(dates_df))

class TextSummaryStatEncoder(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        length = pd.Series(X.apply(lambda x: len(str(x))))
        wc = pd.Series(X.apply(lambda x: len(str(x).split(' '))))

        return pd.concat([length, wc], axis=1)

# text_sum_test = train.loc[0:1, ['project_essay_1', 'project_essay_1_len', 'project_essay_1_wc',
#                                 'project_essay_2', 'project_essay_2_len', 'project_essay_2_wc']]
#
# mapper_text_sum = DataFrameMapper([
#     ('project_essay_1', TextSummaryStatEncoder()),
#     ('project_essay_2', TextSummaryStatEncoder()),
#     ('project_essay_1_len', None),
#     ('project_essay_1_wc', None),
#     ('project_essay_2_len', None),
#     ('project_essay_2_wc', None)
#     ], input_df=True)
#
# print(mapper_text_sum.fit_transform(text_sum_test))