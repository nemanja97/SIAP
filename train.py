from decouple import config
from torch.utils.data import DataLoader

from models.Encoder import Encoder
from utils.CocoDataLoader import CocoDataLoader
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

    train_data_loader = CocoDataLoader(dataset=dataset_loader.train_dataset, padding_idx=3, batch_size=32, shuffle=True)
    test_data_loader = CocoDataLoader(dataset=dataset_loader.test_dataset, padding_idx=3)

    # Create encoder and set it to training mode
    encoder = Encoder(embedding_size=256)
    encoder = encoder.to(DEVICE)
    encoder.cuda() if DEVICE == "cuda" else None
    encoder.train()

    for epoch in range(EPOCHS):
        for i, (image, caption) in enumerate(train_data_loader):

            # Give each tensor to selected device
            image = image.to(DEVICE)
            image = image.cuda() if DEVICE == "cuda" else image

            caption = caption.to(DEVICE)
            caption = caption.cuda() if DEVICE == "cuda" else caption

            print(image)
            print(caption)
