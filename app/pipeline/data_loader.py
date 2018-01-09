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
        BROAD_CATEGORY_INDEX_POSITION = 0
        BROAD_CATEGORY = '4' # 1: Coarse genre, 2: Included/forwarded information, 3: Primary topics, 4: Emotional tone
        SUBCATEGORY_INDEX_POSITION = 1

        simple_categories = {}
        for key, categories_for_single_file in categories.items():
            simple_categories[key] = []

            for line in categories_for_single_file.split('\n'):
                line_labels = line.split(',')
                if line_labels[0] != '' and line_labels[BROAD_CATEGORY_INDEX_POSITION] == BROAD_CATEGORY:
                    simple_categories[key].append(line_labels[SUBCATEGORY_INDEX_POSITION])

            simple_categories[key] = list(set(simple_categories[key]))
        return simple_categories


