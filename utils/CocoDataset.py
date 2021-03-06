from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CocoDataset(Dataset):

    def __init__(self):
        super(CocoDataset, self).__init__()
        self.data = defaultdict(lambda: {"image_path": None, "captions": []})
        self.transform = transforms.Compose([
            transforms.Resize(299),  # Required - https://pytorch.org/hub/pytorch_vision_inception_v3/
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Required - https://pytorch.org/hub/pytorch_vision_inception_v3/
        ])
        self.__keys = None

    def get_image_path(self, image_id):
        return self.data[image_id]["image_path"]

    def get_captions(self, image_id):
        return self.data[image_id]["captions"]

    def set_image_path(self, image_id, image_path):
        self.data[image_id]["image_path"] = image_path

    def set_captions(self, image_id, captions):
        self.data[image_id]["captions"] = captions

    def append_caption(self, image_id, caption):
        self.data[image_id]["captions"].append(caption)

    def __getitem__(self, item):
        if self.__keys is None:
            self.__keys = list(self.data.keys())

        return self.__transform_image(item), self.data[self.__keys[item]]["captions"]

    def __len__(self):
        return len(self.data)

    def __transform_image(self, item):
        image = Image.open(self.data[self.__keys[item]]["image_path"]).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor
