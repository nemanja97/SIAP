import sys
from collections import defaultdict

import torch.backends.cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.EncoderDecoder import EncoderDecoder
from utils.CheckpointUtils import save_checkpoint, load_checkpoint
from utils.CocoDataLoader import CocoDataLoader
from utils.CocoDataset import CocoDataset
from utils.Constants import *

if __name__ == "__main__":
    # Prepare data
    train_dataset = CocoDataset(TRAINING_ANNOTATIONS_PATH, TRAINING_IMAGES_FOLDER)
    train_data_loader = CocoDataLoader(dataset=train_dataset, padding_idx=PADDING_INDEX,
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
        vocab_size=len(train_dataset.vocabulary)
    )

    # Prepare loss criteria optimizations
    criterion = CrossEntropyLoss(ignore_index=PADDING_INDEX)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Load model
    step = 0
    if LOAD_MODEL:
        checkpoint = load_checkpoint("coco.big.tar", DEVICE)
        train_dataset.vocabulary.itos = defaultdict(lambda: "<unk>", checkpoint["vocabulary.itos"])
        train_dataset.vocabulary.stoi = defaultdict(lambda: 1, checkpoint["vocabulary.stoi"])

        EMBEDDING_SIZE = checkpoint["embedding_size"]
        HIDDEN_SIZE = checkpoint["hidden_size"]
        GRU_LAYERS = checkpoint["gru_layers"]
        model = EncoderDecoder(
            embedding_size=EMBEDDING_SIZE,
            hidden_size=HIDDEN_SIZE,
            gru_layers=GRU_LAYERS,
            vocab_size=len(train_dataset.vocabulary)
        )

        model.load_state_dict(checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]

    # Prepare training
    model = model.to(DEVICE)
    model.train()
    writer = SummaryWriter(LOGS_SAVE_LOCATION)

    for epoch in range(EPOCHS):
        save_checkpoint({
            "embedding_size": EMBEDDING_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "gru_layers": GRU_LAYERS,
            "state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vocabulary.itos": dict(train_dataset.vocabulary.itos),
            "vocabulary.stoi": dict(train_dataset.vocabulary.stoi),
            "step": step,
        }, "coco.big.tar")

        for i, (images, captions) in tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=False):
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

