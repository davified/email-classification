from sklearn.model_selection import train_test_split


class VectorizedData:
    def __init__(self, X, y, random_state=0):
        self.X = X
        self.y = y
        (self.X_train, self.X_val, self.y_train, self.y_val) = train_test_split(self.X, self.y, random_state=random_state)

    def get_count(self, y):
        return y.sum(axis=0)
