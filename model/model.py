import torch.nn as nn
from model.AtlasNet import Atlasnet
import model.resnet as resnet

class EncoderDecoder(nn.Module):
    """
    Wrapper for a encoder and a decoder.
    """

    def __init__(self, opt):
        super(EncoderDecoder, self).__init__()
        self.encoder = resnet.resnet18(pretrained=False, num_classes=opt.bottleneck_size)
       
        self.decoder = Atlasnet(opt)
        self.to(opt.device)

        self.eval()

    def forward(self, x, train=True):
        return self.decoder(self.encoder(x), train=train)

    def generate_mesh(self, x):
        return self.decoder.generate_mesh(self.encoder(x))

    def generate_mesh_by_latent(self, x):
        return self.decoder.generate_mesh(x)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)