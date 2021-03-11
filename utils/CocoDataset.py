from collections import defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CocoDataset(Dataset):

    def __init__(self):
        super(CocoDataset, self).__init__()
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((299, 229)),  # Required - https://pytorch.org/hub/pytorch_vision_inception_v3/
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            # Required - https://pytorch.org/hub/pytorch_vision_inception_v3/
        ])
        self.vocabulary = None

    def __getitem__(self, item):
        return self.__transform_image(self.data[item]), self.__transform_captions(self.data[item])

    def __len__(self):
        return len(self.data)

    def __transform_image(self, item):
        image = Image.open(item["image_path"]).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor

    def __transform_captions(self, item):
        numerical_translation = self.vocabulary.translate_caption(item["caption"])
        return torch.LongTensor(numerical_translation)
