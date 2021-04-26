import torch
from torch import nn

from models.Decoder import Decoder
from models.Encoder import Encoder


class EncoderDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, gru_layers):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(embedding_size)
        self.decoder = Decoder(embedding_size, hidden_size, vocab_size, gru_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoder.gru(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                item = predicted.item()
                result_caption.append(item)
                x = self.decoder.embed(predicted).unsqueeze(0)

                if vocabulary.itos[item] == "<end>":
                    break

            return [vocabulary.itos[idx] for idx in result_caption]
