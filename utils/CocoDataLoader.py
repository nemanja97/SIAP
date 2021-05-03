from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


class CocoDataLoader(DataLoader):

    def __init__(self, **kwargs):
        super(CocoDataLoader, self).__init__(
            collate_fn=CocoDataCollate(kwargs.pop("padding_idx")),
            **kwargs
        )


class CocoDataCollate:

    def __init__(self, padding_idx):
        self.__padding_idx = padding_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        captions = [item[1] for item in batch]
        captions = pad_sequence(captions, batch_first=False, padding_value=self.__padding_idx)
        return images, captions
