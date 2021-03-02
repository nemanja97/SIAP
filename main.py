import json
import string
from collections import defaultdict, Counter
from itertools import chain
import numpy as np
import pandas as pd

if __name__ == "__main__":
    train_file = json.load(open("data/annotations/captions_train2017.json"))
    test_file = json.load(open("data/annotations/captions_val2017.json"))

    train_ids = [obj['id'] for obj in train_file['images']]
    test_ids = [obj['id'] for obj in test_file['images']]

    train_file_paths = ["data/train2017/" + obj['file_name'] for obj in train_file['images']]
    test_file_paths = ["data/val2017/" + obj['file_name'] for obj in test_file['images']]

    train_dict = defaultdict(list)
    for obj in train_file['annotations']:
        train_dict[obj["image_id"]].append("<START> " + obj["caption"] + " <END>")

    test_dict = defaultdict(list)
    for obj in test_file['annotations']:
        test_dict[obj["image_id"]].append("<START> " + obj["caption"] + " <END>")

    captions = np.hstack(train_dict.values()).squeeze()
    captions = [caption.lower() for caption in captions]
    captions = [caption.translate(string.punctuation) for caption in captions]
    all_words = set(chain.from_iterable([caption.split() for caption in captions]))

    counter = Counter(all_words)
    most_common_words = counter.most_common(10002)

    vocabulary = set(["<PAD>", "<UNK>"] + [obj[0] for obj in most_common_words])
    words_not_in_vocab = all_words.difference(vocabulary)

    word_replacement_dict = dict()
    for word in words_not_in_vocab:
        word_replacement_dict[word] = "<UNK>"

    for key in train_dict:
        for unknown_word in words_not_in_vocab:
            captions = train_dict[key]
            train_dict[key] = [caption.replace(unknown_word, "<UNK>") for caption in captions]

    print()




