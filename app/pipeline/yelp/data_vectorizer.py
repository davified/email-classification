from sklearn.feature_extraction.text import TfidfVectorizer


class DataVectorizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3)

    def get_vectorized_data(self, texts):
        if type(texts) != list:
            raise Exception('get_vectorized_data() accepts a list as the first argument')
        print('vectorizing data')
        data = self.vectorizer.fit_transform(texts)

        return data
