class DataPipeline:
    def __init__(self):
        pass

    def get_data_and_labels(self, data, labels):

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

        return (data_list, labels_list, filenames_list)
