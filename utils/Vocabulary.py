from collections import defaultdict

import spacy

from nltk import FreqDist
from utils.Constants import NUMBER_OF_WORDS_FOR_VOCABULARY

english_tokenizer = spacy.load("en_core_web_sm")


class Vocabulary:

    def __init__(self, train_captions):
        super(Vocabulary, self).__init__()
        if train_captions is not None:
            self.stoi = self.__prepare_stoi()
            self.itos = self.__prepare_itos()
            self.__prepare_vocabulary(train_captions)

    def translate_caption(self, caption):
        tokens = [token.text.lower() for token in english_tokenizer.tokenizer(caption)]
        return [self.stoi["<start>"]] +\
               [self.stoi[token] for token in tokens] +\
               [self.stoi["<end>"]]

    def __prepare_vocabulary(self, train_captions):
        vocab = FreqDist()
        for caption in train_captions:
            vocab.update([token.text.lower() for token in english_tokenizer.tokenizer(caption)])
        most_common_words = list(map(lambda token: token[0], vocab.most_common(NUMBER_OF_WORDS_FOR_VOCABULARY)))

        idx = 4
        for word in most_common_words:
            if word not in self.stoi.keys():
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    @staticmethod
    def __prepare_stoi():
        stoi = defaultdict(lambda: 1)
        stoi["<start>"] = 0
        stoi["<unk>"] = 1
        stoi["<end>"] = 2
        stoi["<pad>"] = 3
        return stoi

    @staticmethod
    def __prepare_itos():
        itos = defaultdict(lambda: "<unk>")
        itos[0] = "<start>"
        itos[1] = "<unk>"
        itos[2] = "<end>"
        itos[3] = "<pad>"
        return itos

    def __len__(self):
        return len(self.itos)
