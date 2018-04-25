def getCatFeatures(T, Col):

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(binary=True, ngram_range=(1,1), tokenizer=lambda x:x.split(','))
    return vectorizer.fit_transform(T[Col].fillna(''))

