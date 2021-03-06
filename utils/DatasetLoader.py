import json
import string
from os import path

from nltk import FreqDist

from utils.CocoDataset import CocoDataset


class DatasetLoader:

    def __init__(self, train_annotations_path, test_annotations_path, train_data_path, test_data_path):
        self.__train_annotations_path = train_annotations_path
        self.__test_annotations_path = test_annotations_path
        self.__train_data_path = train_data_path
        self.__test_data_path = test_data_path
        self.vocabulary = None
        self.unknown = None
        self.train_dataset = CocoDataset()
        self.test_dataset = CocoDataset()

        self.__load_data()

    def __load_data(self):
        train_data = json.load(open(self.__train_annotations_path))
        test_data = json.load(open(self.__test_annotations_path))

        self.__get_image_paths(train_data, test_data)
        self.__get_captions(train_data, test_data)

    def __get_image_paths(self, train_data, test_data):
        for line in train_data["images"]:
            self.train_dataset.set_image_path(line["id"], path.join(self.__train_data_path, line["file_name"]))

        for line in test_data["images"]:
            self.test_dataset.set_image_path(line["id"], path.join(self.__test_data_path, line["file_name"]))

    def __get_captions(self, train_data, test_data):
        if not self.vocabulary:
            self.__prepare_vocabulary(train_data)

        for row in train_data["annotations"]:
            self.train_dataset.append_caption(row["image_id"], self.__preprocess_caption(row["caption"], True))

        for row in test_data["annotations"]:
            self.test_dataset.append_caption(row["image_id"], self.__preprocess_caption(row["caption"], False))

    def __preprocess_caption(self, sentence, vocabulary: bool):
        caption = self.__clean_caption(sentence)
        if vocabulary:
            caption = " ".join(["<unk>" if word in self.unknown else word for word in caption.split()])

        caption = f"<start> {caption} <end>"
        return caption

    def __get_captions_list(self, train_data):
        return map(
            lambda row: self.__preprocess_caption(row["caption"], False),
            train_data["annotations"])

    def __prepare_vocabulary(self, train_data):
        vocab = FreqDist()

        for caption in self.__get_captions_list(train_data):
            vocab.update(caption.split())

        self.vocabulary = set(map(lambda token: token[0], vocab.most_common(10002)))
        self.unknown = set(map(lambda token: token[0], vocab.items())) - self.vocabulary

    @staticmethod
    def __clean_caption(sentence):
        return sentence.lower().translate(str.maketrans("", "", string.punctuation))

