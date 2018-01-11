import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

app_dir = os.path.dirname(os.path.abspath(__file__))
models_cache_dir = os.path.join(app_dir, '../../models_cache')
vectorized_data_filepath=os.path.join(models_cache_dir, 'yelp_reviews.npy')


class DataVectorizer:
    def __init__(self, texts, load_from_file=False):
        if type(texts) != list:
            raise Exception('get_vectorized_data() accepts a list as the first argument')
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
        print('vectorizing data')

        if load_from_file:
            self.data = np.load(vectorized_data_filepath)
        else:
            data = self.vectorizer.fit_transform(texts)
            np.save(vectorized_data_filepath, data)
            self.data = data
