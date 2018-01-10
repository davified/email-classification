import json


class DataLoader:
    def __init__(self, source_dir):
        self.source_dir = source_dir

    def get_data_and_labels(self):
        reviews = self._load_data("yelp_FILL_IN_PROPER_NAME")
        texts = self._get_texts(reviews)

        binstars = [0 if review['stars'] <= 3 else 1 for review in reviews]
        balanced_texts = []
        balanced_labels = []
        limit = 100000  # Change this to grow/shrink the dataset
        neg_pos_counts = [0, 0]
        for i in range(len(texts)):
            polarity = binstars[i]
            if neg_pos_counts[polarity] < limit:
                balanced_texts.append(texts[i])
                balanced_labels.append(binstars[i])
                neg_pos_counts[polarity] += 1

        return balanced_texts, balanced_labels, None

    def _load_data(self, filename):
        with open(filename) as f:
            reviews = f.read().strip().split("\n")
        reviews = [json.loads(review) for review in reviews]
        return reviews

    def _get_texts(self, reviews):
        texts = [review['text'] for review in reviews]
        return texts
