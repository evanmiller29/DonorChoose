def getTimeFeatures(T, dateCol):
    T['year'] = T[dateCol].apply(lambda x: x.year)
    T['month'] = T[dateCol].apply(lambda x: x.month)
    T['day'] = T[dateCol].apply(lambda x: x.day)
    T['dow'] = T[dateCol].apply(lambda x: x.dayofweek)
    T['hour'] = T[dateCol].apply(lambda x: x.hour)
    T['days'] = (T[dateCol]-T[dateCol].min()).apply(lambda x: x.days)
    return T