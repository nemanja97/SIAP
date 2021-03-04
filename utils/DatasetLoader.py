import json
import string
from collections import defaultdict

from nltk import FreqDist

from utils.DatasetPartitionType import DatasetPartitionType


class DatasetAnnotationsLoader:
    def __init__(self, train_path, validation_path):
        self.__train_path = train_path
        self.__validation_path = validation_path
        self.vocabulary = None
        self.unknown = None

    def get_captions(self, data_partition_type: DatasetPartitionType):
        if not self.vocabulary:
            self.__prepare_vocabulary()

        captions = defaultdict(list)

        for row in self.__load_file(data_partition_type)['annotations']:
            captions[row["image_id"]].append(self.__preprocess_caption(row['caption'], True))

        return captions

    def __get_captions_list(self, data_partition_type: DatasetPartitionType):
        return map(
            lambda row: self.__preprocess_caption(row['caption'], False),
            self.__load_file(data_partition_type)["annotations"])

    def __preprocess_caption(self, sentence, vocabulary: bool):
        caption = self.__clean_caption(sentence)
        if vocabulary:
            caption = ' '.join(['<unk>' if word in self.unknown else word for word in caption.split()])

        caption = f"<start> {caption} <end>"
        return caption

    @staticmethod
    def __clean_caption(sentence):
        return sentence.lower().translate(str.maketrans('', '', string.punctuation))

    def __load_file(self, data_partition_type: DatasetPartitionType):
        if data_partition_type == DatasetPartitionType.TRAINING:
            return json.load(open(self.__train_path))
        elif data_partition_type == DatasetPartitionType.VALIDATION:
            return json.load(open(self.__validation_path))
        else:
            raise Exception("Invalid data partition type")

    def __prepare_vocabulary(self):
        vocab = FreqDist()

        for caption in self.__get_captions_list(DatasetPartitionType.TRAINING):
            vocab.update(caption.split())

        self.vocabulary = set(map(lambda token: token[0], vocab.most_common(10002)))
        self.unknown = set(map(lambda token: token[0], vocab.items())) - self.vocabulary
