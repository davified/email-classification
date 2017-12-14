from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import MultiLabelBinarizer


class DataVectorizer:
    def __init__(self, texts, labels, num_words=10000):
        self.texts = texts
        self.labels = labels
        self.tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.tokenizer.fit_on_texts(self.texts)

    def get_vectorized_data_and_labels(self, max_length=1000):
        sequences = self._convert_texts_to_sequences()
        data = self._zeropad_sequences(sequences, max_length=max_length)

        labels = self._convert_labels_to_categorical_vector()

        return data, labels

    def _convert_texts_to_sequences(self):
        sequences = self.tokenizer.texts_to_sequences(self.texts)
        return sequences

    def _zeropad_sequences(self, sequences, max_length=1000):
        return sequence.pad_sequences(sequences, maxlen=max_length)

    def _convert_labels_to_categorical_vector(self):
        return MultiLabelBinarizer().fit_transform(self.labels)
