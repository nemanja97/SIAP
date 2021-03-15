import json
from collections import defaultdict
from os import path

from utils.CocoDataset import CocoDataset
from utils.Vocabulary import Vocabulary


class DatasetLoader:

    def __init__(self, train_annotations_path, test_annotations_path, train_data_path, test_data_path):
        self.__train_annotations_path = train_annotations_path
        self.__test_annotations_path = test_annotations_path
        self.__train_data_path = train_data_path
        self.__test_data_path = test_data_path
        self.__train_data = defaultdict(lambda: {"image_path": None, "captions": []})
        self.__test_data = defaultdict(lambda: {"image_path": None, "captions": []})

        self.vocabulary = None
        self.train_dataset = CocoDataset()
        self.test_dataset = CocoDataset()

        self.__load_data()

    def __load_data(self):
        train_data = json.load(open(self.__train_annotations_path))
        test_data = json.load(open(self.__test_annotations_path))

        self.__get_image_paths(train_data, test_data)
        self.__get_captions(train_data, test_data)
        train_data, test_data = self.__process_data_for_datasets()

        self.train_dataset.data = train_data
        self.test_dataset.data = test_data
        self.train_dataset.vocabulary = self.vocabulary
        self.test_dataset.vocabulary = self.vocabulary

    def __get_image_paths(self, train_data, test_data):
        for line in train_data["images"]:
            self.__train_data[line["id"]]["image_path"] = path.join(self.__train_data_path, line["file_name"])

        for line in test_data["images"]:
            self.__test_data[line["id"]]["image_path"] = path.join(self.__test_data_path, line["file_name"])

    def __get_captions(self, train_data, test_data):
        if self.vocabulary is None:
            self.vocabulary = Vocabulary(train_data)

        for row in train_data["annotations"]:
            self.__train_data[row["image_id"]]["captions"].append(
                self.vocabulary.preprocess_caption(row["caption"], True))

        for row in test_data["annotations"]:
            self.__test_data[row["image_id"]]["captions"].append(
                self.vocabulary.preprocess_caption(row["caption"], False))

    def __process_data_for_datasets(self):
        train_data = list()
        for item in self.__train_data.values():
            for caption in item["captions"]:
                train_data.append({"image_path": item["image_path"], "caption": caption})

        test_data = list()
        for item in self.__test_data.values():
            for caption in item["captions"]:
                test_data.append({"image_path": item["image_path"], "caption": caption})

        return train_data, test_data
