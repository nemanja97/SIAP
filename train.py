import sys

import torch.backends.cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.EncoderDecoder import EncoderDecoder
from utils.CheckpointUtils import save_checkpoint, load_checkpoint
from utils.CocoDataLoader import CocoDataLoader
from utils.Constants import *
from utils.DatasetLoader import DatasetLoader

if __name__ == "__main__":
    # Prepare data
    dataset_loader = DatasetLoader(TRAINING_ANNOTATIONS_PATH, VALIDATION_ANNOTATIONS_PATH,
                                   TRAINING_IMAGES_FOLDER, VALIDATION_IMAGES_FOLDER)
    train_data_loader = CocoDataLoader(dataset=dataset_loader.train_dataset, padding_idx=PADDING_INDEX,
                                       batch_size=BATCH_SIZE, num_workers=4,
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
    step = 0

    # Load model
    if LOAD_MODEL:
        checkpoint = load_checkpoint("SIAP_MODEL.tar", DEVICE)
        model.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]

    # Prepare training
    model.train()
    writer = SummaryWriter(LOGS_SAVE_LOCATION)

    for epoch in range(EPOCHS):
        for i, (images, captions, _) in tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=False):
            # Give each tensor to selected device
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            # Train model
            outputs = model(images, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            # Process loss
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            # Optimize
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        save_checkpoint({
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }, "SIAP_MODEL.tar")
