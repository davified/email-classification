import unittest

from app.pipeline.data_vectorizer import DataVectorizer


class TestDataVectorizer(unittest.TestCase):
    def setUp(self):
        self.vectorizer = DataVectorizer()

        self.texts = ['hello world !', 'hello python???', 'hello, this is python']
        self.labels = ['0', '1', '0']

    def test_should_return_tokenizer(self):
        self.vectorizer.fit_tokenizer(self.texts)

        self.assertEqual(self.vectorizer.tokenizer.document_count, 3)
        self.assertEqual(self.vectorizer.tokenizer.word_docs['hello'], 3)
        self.assertNotIn('!', self.vectorizer.tokenizer.word_docs.keys())
        self.assertNotIn('?', self.vectorizer.tokenizer.word_docs.keys())
        self.assertNotIn(',', self.vectorizer.tokenizer.word_docs.keys())

    def test_should_convert_text_to_sequences(self):
        self.vectorizer.fit_tokenizer(self.texts)

        sequences = self.vectorizer.convert_texts_to_sequences(self.texts)
        self.assertEquals(sequences, [[1, 3], [1, 2], [1, 4, 5, 2]])

    def test_should_zeropad_sequences(self):
        self.vectorizer.fit_tokenizer(self.texts)
        sequences = self.vectorizer.convert_texts_to_sequences(self.texts)
        MAX_SEQUENCE_LENGTH = 150
        padded_sequences = self.vectorizer.zeropad_sequences(sequences, max_length=MAX_SEQUENCE_LENGTH)

        self.assertEquals(padded_sequences.shape, (len(sequences), MAX_SEQUENCE_LENGTH))

    def test_should_convert_labels_to_categorical_vector(self):
        labels = self.vectorizer.convert_labels_to_categorical_vector(self.labels)

        self.assertEquals(labels.shape, (len(self.labels), len(set(self.labels))))
