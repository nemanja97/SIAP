from tqdm import tqdm

from models.EncoderDecoder import EncoderDecoder
from utils.CheckpointUtils import load_checkpoint
from utils.CocoDataLoader import CocoDataLoader
from utils.DatasetLoader import DatasetLoader
from utils.Constants import *

from torchtext.data.metrics import bleu_score


def try_model(file_name):
    checkpoint = load_checkpoint(file_name, 'cpu')

    dataset_loader = DatasetLoader(TRAINING_ANNOTATIONS_PATH, VALIDATION_ANNOTATIONS_PATH,
                                   TRAINING_IMAGES_FOLDER, VALIDATION_IMAGES_FOLDER)

    model = EncoderDecoder(embedding_size=EMBEDDING_SIZE,
                           hidden_size=HIDDEN_SIZE,
                           gru_layers=GRU_LAYERS,
                           vocab_size=len(dataset_loader.train_dataset.vocabulary))

    model.load_state_dict(checkpoint['state'])
    model.eval()

    test_data_loader = CocoDataLoader(dataset=dataset_loader.test_dataset, shuffle=False, pin_memory=True,
                                      padding_idx=PADDING_INDEX)

    predicted = []
    true = []
    for i, (image, caption, image_path) in tqdm(enumerate(test_data_loader), total=len(test_data_loader), leave=False):
        # add break for testing
        pred = model.caption_image(image, dataset_loader.test_dataset.vocabulary)
        print(pred)
        predicted.append(pred)
        print(image_path)
        t = [dataset_loader.test_dataset.vocabulary.itos[idx[0]] for idx in caption.tolist()]
        print(t)
        true.append(t)

    print("BLEU_SCORE: ", bleu_score(predicted, true))


if __name__ == '__main__':
    try_model("SIAP_MODEL.tar")
