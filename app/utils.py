import os
from collections import OrderedDict


def load_files_as_dict(source_directory, file_ext):
    categories = {}
    for i, file in enumerate(os.listdir(source_directory)):
        if file_ext in file:
            contents = open('{}/{}'.format(source_directory, file), mode='r')
            key = int(file.split('.')[0])
            categories[key] = contents.read()
            contents.close()

    return categories


def sort_dictionary_by_keys(dictionary):
    return OrderedDict(sorted(dictionary.items()))

def simplify_categories(categories):
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
