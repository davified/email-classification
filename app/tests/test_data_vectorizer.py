import unittest

from app.pipeline.data_vectorizer import DataVectorizer


class TestDataVectorizer(unittest.TestCase):
    def setUp(self):
        self.texts = ['hello world !', 'hello python???', 'hello, this is python']
        self.labels = ['0', '1', '0']

        self.vectorizer = DataVectorizer(self.texts, self.labels)

    def test_vectorizer_should_fit_tokenizer_upon_instantiation(self):
        self.assertEqual(self.vectorizer.tokenizer.document_count, 3)
        self.assertEqual(self.vectorizer.tokenizer.word_docs['hello'], 3)
        self.assertNotIn('!', self.vectorizer.tokenizer.word_docs.keys())
        self.assertNotIn('?', self.vectorizer.tokenizer.word_docs.keys())
        self.assertNotIn(',', self.vectorizer.tokenizer.word_docs.keys())

    def test_should_convert_text_to_sequences(self):
        sequences = self.vectorizer._convert_texts_to_sequences()
        self.assertEqual(sequences, [[1, 3], [1, 2], [1, 4, 5, 2]])

    def test_should_zeropad_sequences(self):
        MAX_SEQUENCE_LENGTH = 150
        sequences = self.vectorizer._convert_texts_to_sequences()
        padded_sequences = self.vectorizer._zeropad_sequences(sequences, max_length=MAX_SEQUENCE_LENGTH)

        self.assertEqual(padded_sequences.shape, (len(sequences), MAX_SEQUENCE_LENGTH))

    def test_should_convert_labels_to_categorical_vector(self):
        labels = self.vectorizer._convert_labels_to_categorical_vector()

        self.assertEqual(labels.shape, (len(self.labels), len(set(self.labels))))

    def test_should_return_vectorized_data_and_labels(self):
        MAX_LENGTH=42
        data, labels = self.vectorizer.get_vectorized_data_and_labels(max_length=MAX_LENGTH)
        self.assertEqual(data.shape, (len(self.texts), MAX_LENGTH))
        self.assertEqual(labels.shape, (len(self.labels), len(set(self.labels))))
