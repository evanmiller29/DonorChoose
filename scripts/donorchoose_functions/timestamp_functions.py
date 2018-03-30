def extract_timestamp_features(df):

    """

    :param df: Dataframe that holds the timestamp information
    :return: A dataframe with additional time stamp information
    """

    import pandas as pd
    import numpy as np

    df['subtime_year'] = df['project_submitted_datetime'].apply(lambda x: int(x.split('-')[0]))
    df['subtime_month'] = df['project_submitted_datetime'].apply(lambda x: int(x.split('-')[1]))
    df['subtime_date'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[0].split('-')[2]))
    df['subtime_day_of_week'] = pd.to_datetime(df['project_submitted_datetime']).dt.weekday
    df['subtime_hour'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[-1].split(':')[0]))
    df['subtime_minute'] = df['project_submitted_datetime'].apply(lambda x: int(x.split(' ')[-1].split(':')[1]))
    df['subtime_project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime']).values.astype(np.int64)

    return df
