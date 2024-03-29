import string
from collections import defaultdict
from os.path import join

import pandas as pd
from torchtext.data.metrics import bleu_score

from models.EncoderDecoder import EncoderDecoder
from utils.CheckpointUtils import load_checkpoint
from utils.Constants import *
from utils.ImageTransormation import transform
from utils.Vocabulary import Vocabulary


def try_model(file_name):
    # Load testing data
    test_dataframe = pd.read_csv(VALIDATION_ANNOTATIONS_PATH)

    # Load model and vocabulary
    checkpoint = load_checkpoint(file_name, 'cpu')
    vocabulary = load_vocabulary(checkpoint)
    model = load_model(checkpoint, vocabulary)
    model.eval()

    predicted = []
    true = []

    for image_name in test_dataframe.image.unique():
        image = transform(join(VALIDATION_IMAGES_FOLDER, image_name)).unsqueeze(0)
        predicted_caption = caption_image(image, model, vocabulary)

        print("OUTPUT: " + predicted_caption)

        true_captions_for_image = []
        for _, row in test_dataframe.loc[test_dataframe['image'] == image_name].iterrows():
            caption = row["caption"]
            print("CORRECT: " + caption)
            true_captions_for_image.append(caption.translate(
                str.maketrans("", "", string.punctuation)
            ).split())
        print("-------------------------------------------------------------------------------------------------------")

        true.append(
            true_captions_for_image
        )
        predicted.append(
            predicted_caption.translate(
                str.maketrans("", "", string.punctuation)
            ).split()
        )

    print("BLEU_SCORE: ", bleu_score(predicted, true))


def caption_image(image, model, vocabulary):
    captioned_image = model.caption_image(image, vocabulary)
    captioned_image_without_tokens = captioned_image[1:-1]  # Strip <start> and <end>

    sentence = " ".join(captioned_image_without_tokens)
    return sentence.capitalize()


def load_vocabulary(checkpoint):
    vocabulary = Vocabulary(None)
    vocabulary.stoi = defaultdict(lambda: 1, checkpoint["vocabulary.stoi"])
    vocabulary.itos = defaultdict(lambda: "<unk>", checkpoint["vocabulary.itos"])
    return vocabulary


def load_model(checkpoint, vocabulary):
    model = EncoderDecoder(
        embedding_size=checkpoint["embedding_size"],
        hidden_size=checkpoint["hidden_size"],
        gru_layers=checkpoint["gru_layers"],
        vocab_size=len(vocabulary)
    )
    model.load_state_dict(checkpoint["state"])
    return model


if __name__ == '__main__':
    try_model("coco.big.tar")
