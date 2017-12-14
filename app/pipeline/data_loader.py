import os
from collections import OrderedDict


class DataLoader:
    def __init__(self, source_dir):
        self.source_dir = source_dir

    def get_data_and_labels(self):
        data, labels = self._load_and_sort_data(self.source_dir)
        data_list = []
        labels_list = []
        filenames_list = []

        if len(data) != len(labels):
            raise Exception('data and labels are of differing length')

        for k, v in data.items():
            data_list.append(v)

        for k, v in labels.items():
            labels_list.append(v)

        for i in range(len(labels)):
            data_key = list(data.items())[i][0]
            label_key = list(labels.items())[i][0]
            if data_key == label_key:
                filenames_list.append(data_key)
            else:
                raise Exception('data and labels are not sorted in sequence')

        return data_list, labels_list, filenames_list

    def _load_and_sort_data(self, source_dir):
        categories = self._load_files_as_dict(source_dir, '.cats')
        simple_categories = self._simplify_categories(categories)
        sorted_categories = self._sort_dictionary_by_keys(simple_categories)
        emails = self._load_files_as_dict(source_dir, '.txt')
        sorted_emails = self._sort_dictionary_by_keys(emails)
        return sorted_emails, sorted_categories

    def _load_files_as_dict(self, source_directory, file_ext):
        categories = {}
        for i, file in enumerate(os.listdir(source_directory)):
            if file_ext in file:
                contents = open('{}/{}'.format(source_directory, file), mode='r')
                key = int(file.split('.')[0])
                categories[key] = contents.read()
                contents.close()

        return categories

    def _sort_dictionary_by_keys(self, dictionary):
        return OrderedDict(sorted(dictionary.items()))

    def _simplify_categories(self, categories):
        # a simple/downsampled implementation of category-labelling for the first iteration
        simple_categories = {}
        for key, category in categories.items():
            simple_categories[key] = []
            categories_in_a_single_email = category.split('\n')

            for cat in categories_in_a_single_email:
                if cat.split(',')[0] != '':
                    simple_categories[key].append(cat.split(',')[0])

            simple_categories[key] = list(set(simple_categories[key]))

        return simple_categories


