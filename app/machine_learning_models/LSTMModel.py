import numpy as np
from sklearn import metrics
from keras.layers import Embedding, Dropout, Conv1D, MaxPooling1D, Dense, LSTM
from keras.models import Sequential


class LSTMModel:
    def __init__(self, input_length, no_of_outputs):
        self.classifier = Sequential()
        self.classifier.add(Embedding(20000, 128, input_length=input_length))
        self.classifier.add(Dropout(0.2))
        self.classifier.add(Conv1D(64, 5, activation='relu'))
        self.classifier.add(MaxPooling1D(pool_size=4))
        self.classifier.add(LSTM(128))
        self.classifier.add(Dense(no_of_outputs, activation='sigmoid'))
        self.classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=3):
        self.classifier.fit(X_train, np.array(y_train), validation_split=0.5, epochs=epochs)

    def predict(self, X):
        return self.classifier.predict(X).argmax(axis=1)

    def _calculate_recall_score(self, X_val, y_val):
        y_predicted = self.classifier.predict(X_val)
        return metrics.recall_score(y_val, y_predicted.argmax(axis=1), average='weighted')

    def _calculate_precision_score(self, X_val, y_val):
        y_predicted = self.classifier.predict(X_val)
        return metrics.precision_score(y_val, y_predicted, average='weighted')



