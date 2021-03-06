def extract_text_features(df):

    """

    :param df: Data frame for extracting length of text fields
    :return: Dataframe with additional columns relating to
    """

    import pandas as pd

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

    return df


def join_essays(df):

    """

    :param df: Dataframe that holds all the essays
    :return: A series that includes all the essays joined into one
    """

    essay = df.apply(lambda x: ' '.join([
    str(x['project_essay_1']),
    str(x['project_essay_2']),
    str(x['project_essay_3']),
    str(x['project_essay_4']),
    ]), axis=1)

    return essay


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