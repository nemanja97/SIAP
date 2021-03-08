import string

from nltk import FreqDist


class Vocabulary:

    def __init__(self, train_data):
        super(Vocabulary, self).__init__()
        self.known = None
        self.unknown = None
        self.itos = {0: "<start>", 1: "<unk>", 2: "<end>", 3: "<pad>"}
        self.stoi = {"<start>": 0, "<unk>": 1, "<end>": 2, "<pad>": 3}
        self.__prepare_vocabulary(train_data)

    def preprocess_caption(self, sentence, vocabulary: bool):
        caption = self.__clean_caption(sentence)
        if vocabulary:
            caption = " ".join(["<unk>" if word in self.unknown else word for word in caption.split()])

        caption = f"<start> {caption} <end>"
        return caption

    def translate_caption(self, caption):
        return [self.stoi["<unk>"] if word not in self.stoi else self.stoi[word] for word in caption.split()]

    def __prepare_vocabulary(self, train_data):
        vocab = FreqDist()

        for caption in self.__get_captions_list(train_data):
            vocab.update(caption.split())

        vocabulary = set(map(lambda token: token[0], vocab.most_common(10002)))
        vocabulary.add("<pad>")

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

    @staticmethod
    def __clean_caption(sentence):
        return sentence.lower().translate(str.maketrans("", "", string.punctuation))
