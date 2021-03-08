from decouple import config
from torch.utils.data import DataLoader

from models.Encoder import Encoder
from utils.DatasetLoader import DatasetLoader

TRAINING_ANNOTATIONS_PATH = config("TRAINING_ANNOTATIONS_PATH")
VALIDATION_ANNOTATIONS_PATH = config("VALIDATION_ANNOTATIONS_PATH")
TRAINING_IMAGES_FOLDER = config("TRAINING_IMAGES_FOLDER")
VALIDATION_IMAGES_FOLDER = config("VALIDATION_IMAGES_FOLDER")
EPOCHS = int(config("EPOCHS"))
DEVICE = config("DEVICE")

if __name__ == "__main__":
    dataset_loader = DatasetLoader(TRAINING_ANNOTATIONS_PATH, VALIDATION_ANNOTATIONS_PATH,
                                   TRAINING_IMAGES_FOLDER, VALIDATION_IMAGES_FOLDER)
    train_data_loader = DataLoader(dataset_loader.train_dataset, shuffle=True)
    test_data_loader = DataLoader(dataset_loader.test_dataset)

    # Create encoder and set it to training mode
    encoder = Encoder(embedding_size=256)
    encoder = encoder.to(DEVICE)
    encoder.cuda() if DEVICE == "GPU" else None
    encoder.train()

    for epoch in range(EPOCHS):
        for i, (image, captions) in enumerate(train_data_loader):
            # Give tensor to GPU
            image = image.to(DEVICE)
            image = image.cuda() if DEVICE == "GPU" else image

            image_features = encoder(image)

            print(image_features)
            print(len(captions))
            print(captions)
