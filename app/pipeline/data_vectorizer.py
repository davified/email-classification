from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import MultiLabelBinarizer


class DataVectorizer:
    def __init__(self, MAX_NB_WORDS=10000):
        self.tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

    def fit_tokenizer(self, texts):
        self.tokenizer.fit_on_texts(texts)

    def convert_texts_to_sequences(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return sequences

    def zeropad_sequences(self, sequences, max_length=1000):
        return sequence.pad_sequences(sequences, maxlen=max_length)

    def convert_labels_to_categorical_vector(self, labels):
        return MultiLabelBinarizer().fit_transform(labels)
