import torch
from decouple import config
from torch.utils.data import DataLoader

from models.Encoder import Encoder
from utils.DatasetLoader import DatasetLoader

TRAINING_ANNOTATIONS_PATH = config("TRAINING_ANNOTATIONS_PATH")
VALIDATION_ANNOTATIONS_PATH = config("VALIDATION_ANNOTATIONS_PATH")
TRAINING_IMAGES_FOLDER = config("TRAINING_IMAGES_FOLDER")
VALIDATION_IMAGES_FOLDER = config("VALIDATION_IMAGES_FOLDER")
EPOCHS = int(config("EPOCHS"))

if __name__ == "__main__":
    dataset_loader = DatasetLoader(TRAINING_ANNOTATIONS_PATH, VALIDATION_ANNOTATIONS_PATH,
                                   TRAINING_IMAGES_FOLDER, VALIDATION_IMAGES_FOLDER)
    train_data_loader = DataLoader(dataset_loader.train_dataset, shuffle=True)
    test_data_loader = DataLoader(dataset_loader.test_dataset)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Create encoder and set it to training mode
    encoder = Encoder(embedding_size=256)
    encoder = encoder.to(device)
    encoder.cuda()
    encoder.train()

    for epoch in range(EPOCHS):
        for i, (image, captions) in enumerate(train_data_loader):
            # Give tensor to GPU
            image = image.to(device)
            image = image.cuda()

            image_features = encoder(image)

            print(image_features)
            print(len(captions))
            print(captions)
