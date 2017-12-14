class DataLoader:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def get_data_and_labels(self):
        data_list = []
        labels_list = []
        filenames_list = []

        if len(self.data) != len(self.labels):
            raise Exception('data and labels are of differing length')

        for k, v in self.data.items():
            data_list.append(v)

        for k, v in self.labels.items():
            labels_list.append(v)

        for i in range(len(self.labels)):
            data_key = list(self.data.items())[i][0]
            label_key = list(self.labels.items())[i][0]
            if data_key == label_key:
                filenames_list.append(data_key)
            else:
                raise Exception('self.data and labels are not sorted in sequence')

        return data_list, labels_list, filenames_list
