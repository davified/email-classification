import os


def load_files_in_directory(source_directory, file_ext):
    categories = {}
    for i, file in enumerate(os.listdir(source_directory)):
        if file_ext in file:
            contents = open('{}/{}'.format(source_directory, file), mode='r')
            key = int(file.split('.')[0])
            categories[key] = contents.read()

    return categories
