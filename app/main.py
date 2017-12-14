import os

from app.VectorizedData.vectorized_data import VectorizedData
from app.pipeline.data_vectorizer import DataVectorizer
from app.utils import load_files_as_dict, sort_dictionary_by_keys, simplify_categories
from app.pipeline.data_loader import DataLoader

os.environ['KERAS_BACKEND'] = 'tensorflow'


def main():
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, '../data/enron_with_categories/1')

    categories = load_files_as_dict(source_dir, '.cats')
    simple_categories = simplify_categories(categories)
    sorted_categories = sort_dictionary_by_keys(simple_categories)

    emails = load_files_as_dict(source_dir, '.txt')
    sorted_emails = sort_dictionary_by_keys(emails)

    data_loader = DataLoader(sorted_emails, sorted_categories)
    data, labels, filenames = data_loader.get_data_and_labels()

    vectorizer = DataVectorizer()
    vectorizer.fit_tokenizer(data)
    sequences = vectorizer.convert_texts_to_sequences(data)
    data_2 = vectorizer.zeropad_sequences(sequences)

    labels_2 = vectorizer.convert_labels_to_categorical_vector(labels)

    vectorized_data = VectorizedData(data_2, labels_2)


if __name__ == "__main__":
    main()
