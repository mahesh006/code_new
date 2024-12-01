import torch
from torchvision import models, transforms

class EfficientNetB4Encoder(torch.nn.Module):
    def __init__(self):
        super(EfficientNetB4Encoder, self).__init__()
        self.encoder = models.efficientnet_b4(pretrained=True)
        self.encoder = torch.nn.Sequential(*(list(self.encoder.children())[:-1]))  # Remove classification head

    def forward(self, images):
        return self.encoder(images).squeeze(-1).squeeze(-1)  # Global feature vector
