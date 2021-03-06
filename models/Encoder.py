from torch import nn
from torchvision.models.inception import inception_v3


class Encoder(nn.Module):

    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.inception = inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))
