import json

from pandas import DataFrame
from utils.Constants import *


def generate_caption(data, save_location):
    ids_to_path_converter = {}
    for image in data["images"]:
        ids_to_path_converter[image["id"]] = image["file_name"]

    image_paths, image_captions = [], []
    for image in data["annotations"]:
        image_paths.append(ids_to_path_converter[image["image_id"]])
        image_captions.append(image["caption"])

    dataframe = DataFrame(list(zip(image_paths, image_captions)), columns=["image", "caption"])
    dataframe.to_csv(save_location, index=False)


if __name__ == "__main__":
    train_data = json.load(open(TRAINING_ANNOTATIONS_PATH))
    test_data = json.load(open(VALIDATION_ANNOTATIONS_PATH))

    generate_caption(train_data, "data/captions/train.csv")
    generate_caption(test_data, "data/captions/test.csv")
