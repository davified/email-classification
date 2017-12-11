import os
from app.utils import load_files_as_dict, sort_dictionary_by_keys, simplify_categories


def main():
    project_dir = os.getcwd()
    source_dir = os.path.join(project_dir, 'data/enron_with_categories/1')

    categories = load_files_as_dict(source_dir, '.cats')
    simple_categories = simplify_categories(categories)
    sorted_categories = sort_dictionary_by_keys(simple_categories)

    emails = load_files_as_dict(source_dir, '.txt')
    sorted_emails = sort_dictionary_by_keys(emails)


if __name__ == "__main__": main()