from torch import nn
from torchvision.models.inception import inception_v3


class Encoder(nn.Module):

    def __init__(self, embedding_size):
        super(Encoder, self).__init__()
        self.inception = self.__prepare_inception_net(embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        return self.dropout(self.relu(features))

    def __prepare_inception_net(self, embedding_size):
        inception = inception_v3(pretrained=True, aux_logits=False)
        inception.fc = nn.Linear(inception.fc.in_features, embedding_size)
        for name, param in inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        return inception
