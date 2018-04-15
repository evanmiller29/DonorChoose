def price_quantity_agg(x):
    """

    :param x: Dataframe that you want to add more aggregations to
    :return: Series that summarises the price / quantity data
    """

    import pandas as pd
    import numpy as np

    names = {
        'price_count': x['price'].count(),
        'price_sum': x['price'].sum(),
        'price_min': x['price'].min(),
        'price_max': x['price'].max(),
        'price_range': x['price'].max() - x['price'].min(),
        'price_mean': x['price'].mean(),
        'price_unique': len(np.unique(x['price'])),
        'quantity_sum': x['quantity'].sum(),
        'quantity_min': x['quantity'].min(),
        'quantity_max': x['quantity'].max(),
        'quantity_mean': x['quantity'].mean(),
        'description_ttl': ', '.join(x['description'].astype(str))

        }

    return pd.Series(names, index=['price_count','price_sum','price_min',
                                   'price_max','price_range','price_mean',
                                   'price_unique','quantity_sum','quantity_min',
                                   'quantity_max','quantity_mean', 'description_ttl'])
