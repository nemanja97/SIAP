import os
import sys
from datetime import datetime

import torch.backends.cudnn
from decouple import config
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm

from models.EncoderDecoder import EncoderDecoder
from utils.CocoDataLoader import CocoDataLoader
from utils.DatasetLoader import DatasetLoader

TRAINING_ANNOTATIONS_PATH = config("TRAINING_ANNOTATIONS_PATH")
VALIDATION_ANNOTATIONS_PATH = config("VALIDATION_ANNOTATIONS_PATH")
TRAINING_IMAGES_FOLDER = config("TRAINING_IMAGES_FOLDER")
VALIDATION_IMAGES_FOLDER = config("VALIDATION_IMAGES_FOLDER")
MODELS_SAVE_LOCATION = config("MODELS_SAVE_LOCATION")
LOGS_SAVE_LOCATION = config("LOGS_SAVE_LOCATION")

DEVICE = config("DEVICE")
EPOCHS = int(config("EPOCHS"))
BATCH_SIZE = int(config("BATCH_SIZE"))

EMBEDDING_SIZE = int(config("EMBEDDING_SIZE"))
HIDDEN_SIZE = int(config("HIDDEN_SIZE"))
GRU_LAYERS = int(config("GRU_LAYERS"))
LEARNING_RATE = float(config("LEARNING_RATE"))

PADDING_INDEX = 3


def save_checkpoint(checkpoint, file_name):
    torch.save(checkpoint, os.path.join(MODELS_SAVE_LOCATION, file_name))


def load_checkpoint(file_name):
    return torch.load(os.path.join(MODELS_SAVE_LOCATION, file_name))


if __name__ == "__main__":
    # Prepare data
    dataset_loader = DatasetLoader(TRAINING_ANNOTATIONS_PATH, VALIDATION_ANNOTATIONS_PATH,
                                   TRAINING_IMAGES_FOLDER, VALIDATION_IMAGES_FOLDER)
    train_data_loader = CocoDataLoader(dataset=dataset_loader.train_dataset, padding_idx=PADDING_INDEX,
                                       batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       pin_memory=True)

    # Prepare torch
    torch.backends.cudnn.benchmark = True

    # Prepare model
    model = EncoderDecoder(
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        gru_layers=GRU_LAYERS,
        vocab_size=len(dataset_loader.train_dataset.vocabulary)
    )
    model = model.to(DEVICE)

    # Prepare loss criteria optimizations
    criterion = CrossEntropyLoss(ignore_index=PADDING_INDEX)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Prepare training
    model.train()
    best_loss_result = sys.maxsize

    for epoch in range(EPOCHS):
        epoch_loss = sys.maxsize
        for i, (images, captions) in tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=False):
            # Give each tensor to selected device
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            # Train model
            outputs = model(images, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            # Optimize
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            print(" Epoch - {}, Time - {}, Loss - {} ".format(epoch, datetime.now().strftime(
                '%H:%M:%S'), loss.item()))
            epoch_loss = loss.item()
        save_checkpoint({
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "loss": epoch_loss
        }, "SIAP_MODEL.tar".format(epoch_loss))
