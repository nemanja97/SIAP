import string
from collections import defaultdict

from nltk import FreqDist


class Vocabulary:

    def __init__(self, train_data):
        super(Vocabulary, self).__init__()
        self.known = None
        self.unknown = None
        self.itos = self.__prepare_itos()
        self.stoi = self.__prepare_stoi()
        self.__prepare_vocabulary(train_data)

    def preprocess_caption(self, sentence, vocabulary: bool):
        caption = self.__clean_caption(sentence)
        if vocabulary:
            caption = " ".join(["<unk>" if word in self.unknown else word for word in caption.split()])

        caption = f"<start> {caption} <end>"
        return caption

    def translate_caption(self, caption):
        return [self.stoi[word] for word in caption.split()]

    def __prepare_vocabulary(self, train_data):
        vocab = FreqDist()

        for caption in self.__get_captions_list(train_data):
            vocab.update(caption.split())

        vocabulary = set(map(lambda token: token[0], vocab.most_common(10002)))

        self.known = vocabulary
        self.unknown = set(map(lambda token: token[0], vocab.items())) - self.known

        idx = 4
        for word in self.known:
            if word not in self.stoi.keys():
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def __get_captions_list(self, train_data):
        return map(
            lambda row: self.preprocess_caption(row["caption"], False),
            train_data["annotations"])

    def __prepare_stoi(self):
        stoi = defaultdict(lambda: 1)
        stoi["<start>"] = 0
        stoi["<unk>"] = 1
        stoi["<end>"] = 2
        stoi["<pad>"] = 3
        return stoi

    def __prepare_itos(self):
        itos = defaultdict(lambda: "<unk>")
        itos[0] = "<start>"
        itos[1] = "<unk>"
        itos[2] = "<end>"
        itos[3] = "<pad>"
        return itos

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def __clean_caption(sentence):
        return sentence.lower().translate(str.maketrans("", "", string.punctuation))
