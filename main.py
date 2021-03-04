from decouple import config

from utils.DatasetLoader import DatasetAnnotationsLoader
from utils.DatasetPartitionType import DatasetPartitionType

TRAINING_ANNOTATIONS_PATH = config("TRAINING_ANNOTATIONS_PATH")
VALIDATION_ANNOTATIONS_PATH = config("VALIDATION_ANNOTATIONS_PATH")

if __name__ == "__main__":
    dataset_loader = DatasetAnnotationsLoader(TRAINING_ANNOTATIONS_PATH, VALIDATION_ANNOTATIONS_PATH)
    print(dataset_loader.get_captions(DatasetPartitionType.TRAINING))
