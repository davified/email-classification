from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import MultiLabelBinarizer

from app.constants import INPUT_LENGTH


class DataVectorizer:
    def __init__(self, texts, labels, num_words=10000):
        self.labels = labels
        self.tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.tokenizer.fit_on_texts(texts)

    def get_vectorized_data(self, texts, max_length=INPUT_LENGTH):
        if type(texts) != list:
            raise Exception('get_vectorized_data() accepts a list as the first argument')
        sequences = self._convert_texts_to_sequences(texts)
        data = self._zeropad_sequences(sequences, max_length=max_length)

        return data

    def get_vectorized_labels(self):
        labels = self._convert_labels_to_categorical_vector()

        return labels

    def _convert_texts_to_sequences(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences

    def _zeropad_sequences(self, sequences, max_length=1000):
        return sequence.pad_sequences(sequences, maxlen=max_length)

    def _convert_labels_to_categorical_vector(self):
        return MultiLabelBinarizer().fit_transform(self.labels)
