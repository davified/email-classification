from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


class CustomRandomForestClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier()

    def train(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def predict(self, X):
        return self.classifier.predict(X)

    def _calculate_recall_score(self, X_val, y_val):
        y_predicted = self.classifier.predict(X_val)
        return metrics.recall_score(y_val, y_predicted, average='weighted')
