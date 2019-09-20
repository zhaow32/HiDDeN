import torch.nn as nn
from options import HiDDenConfiguration
from model.conv_bn_relu import ConvBNRelu


class Decoder(nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, config: HiDDenConfiguration):

        super(Decoder, self).__init__()
        self.channels = config.decoder_channels

        layers = [ConvBNRelu(3, self.channels)]
        for _ in range(config.decoder_blocks - 1):
            layers.append(ConvBNRelu(self.channels, self.channels))

        # layers.append(block_builder(self.channels, config.message_length))
        layers.append(ConvBNRelu(self.channels, config.message_length))
        #get rid of the replicated message information during encoding part
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = nn.Sequential(*layers)
        #这步是什么意思？？？？？？linear????
        #做了一个matrix层面上的linear transform
        self.linear = nn.Linear(config.message_length, config.message_length)

    def forward(self, image_with_wm):
        #image with watermark has size [12,3,128,128]=[batch_size,depth,height,width]
        x = self.layers(image_with_wm)
        #image after convolved is [12,30,1,1]
        x.squeeze_(3).squeeze_(2)
        x = self.linear(x)# x has the size A X 30 in the end
        return x
