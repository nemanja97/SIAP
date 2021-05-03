from os.path import join

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.ImageTransormation import transform
from utils.Vocabulary import Vocabulary


class CocoDataset(Dataset):

    def __init__(self, csv_data_file_location, images_location):
        super(CocoDataset, self).__init__()
        self.dataframe = pd.read_csv(csv_data_file_location)
        self.vocabulary = Vocabulary(self.dataframe["caption"])
        self.images_location = images_location

    def __getitem__(self, item):
        return self.__transform_image(self.dataframe["image"][item]), \
               self.__transform_captions(self.dataframe["caption"][item])

    def __len__(self):
        return len(self.dataframe)

    def __transform_image(self, item):
        return transform(join(self.images_location, item))

    def __transform_captions(self, item):
        numerical_translation = self.vocabulary.translate_caption(item)
        return torch.tensor(numerical_translation)
